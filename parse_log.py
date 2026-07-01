import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. ARGS
# ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default="logs_gridsearch",
                     help="Thư mục chứa các file .log từ run_grid_search.ps1")
parser.add_argument("--metric", default="macro_f1", choices=["macro_f1", "weighted_f1"],
                     help="Metric dùng để xếp hạng config (default: macro_f1, "
                          "khớp với benchmark MVA=0.767 trong paper MOFNet).")
args = parser.parse_args()

# Tên file log do .ps1 sinh ra: {CANCER}_{conv}_{fusion}_lr{LR}_mixup{MIXUP}.log
# LƯU Ý: phần $MIXUP trong .ps1 là "0.0"/"0.1"/"0.2" (dùng dấu chấm), nhưng
# trên thực tế các file log tải lên có tên dạng "..._mixup0_0.log" (dấu
# gạch dưới) -- khả năng do quá trình lưu/đồng bộ file trên Windows đổi
# "." thành "_". Regex dưới đây nhận cả hai kiểu để không bỏ sót file.
FNAME_RE = re.compile(
    r"^(?P<cancer>[A-Z]+)_(?P<conv>gat|gcn)_(?P<fusion>attn|cross)"
    r"_lr(?P<lr>[0-9.eE+-]+)_mixup(?P<mixup>[0-9._]+)\.log$"
)
# Đánh dấu mỗi split bên trong 1 file log: "=== SPLIT SEED: N ===" hoặc
# "=== SPLIT SEED: N -- SKIPPED: ... ==="
SPLIT_RE = re.compile(r"=== SPLIT SEED: (\d+)([^\n]*?)===")


# ─────────────────────────────────────────────
# 1b. ĐỌC FILE LOG (xử lý encoding bị trộn UTF-8 / UTF-16LE)
# ─────────────────────────────────────────────
# NGUYÊN NHÂN LỖI: trong run_grid_search.ps1, các dòng marker
# ("CONFIG: ...", "=== SPLIT SEED: N ===") được ghi bằng
# `Out-File -Encoding utf8` -> UTF-8 thật. Nhưng output huấn luyện của
# python.exe được nối vào cùng file bằng `*>> $LOG_FILE` (dòng 140) --
# và PowerShell luôn redirect output của TIẾN TRÌNH NGOÀI theo encoding
# mặc định của console (UTF-16LE, không BOM) bất kể $env:PYTHONIOENCODING
# hay $OutputEncoding được set thế nào. Kết quả: mỗi file log thực chất
# là UTF-8 và UTF-16LE XEN KẼ NHAU theo từng đoạn ghi. Đọc thẳng bằng
# encoding="utf-8" làm hỏng toàn bộ phần UTF-16LE (chiếm phần lớn file,
# bao gồm cả "FINAL ENSEMBLE RESULT") -> mọi run bị gắn nhãn "incomplete"
# dù đã chạy xong. Hàm dưới đây tách file theo từng marker UTF-8 đã biết
# vị trí chắc chắn, rồi giải mã phần còn lại bằng codec phù hợp.
def smart_read_log(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()

    if raw[:3] == b"\xef\xbb\xbf":  # bỏ BOM nếu có
        raw = raw[3:]

    marker_bytes = "=== SPLIT SEED".encode("utf-8")
    marker_positions = [m.start() for m in re.finditer(re.escape(marker_bytes), raw)]

    def decode_chunk(chunk: bytes) -> str:
        # Đoạn UTF-16LE (do *>> sinh ra) có 1 byte NUL gần như sau mỗi
        # ký tự ASCII; văn bản UTF-8 bình thường thì không. Lấy mẫu để
        # quyết định codec cho từng đoạn, thay vì giả định cả file.
        if not chunk:
            return ""
        sample = chunk[:2000]
        nul_ratio = sample.count(0) / max(len(sample), 1)
        if nul_ratio > 0.3:
            return chunk.decode("utf-16-le", errors="replace")
        return chunk.decode("utf-8", errors="replace")

    if not marker_positions:
        return decode_chunk(raw)

    pieces = [decode_chunk(raw[: marker_positions[0]])]  # phần header đầu file

    for i, p in enumerate(marker_positions):
        # Dòng marker luôn là UTF-8, kết thúc ở CRLF/LF đầu tiên sau nó.
        line_end_crlf = raw.find(b"\r\n", p)
        line_end_lf = raw.find(b"\n", p)
        if line_end_crlf != -1 and (line_end_lf == -1 or line_end_crlf <= line_end_lf):
            line_end = line_end_crlf + 2
        elif line_end_lf != -1:
            line_end = line_end_lf + 1
        else:
            line_end = len(raw)
        pieces.append(raw[p:line_end].decode("utf-8", errors="replace"))

        next_p = marker_positions[i + 1] if i + 1 < len(marker_positions) else len(raw)
        pieces.append(decode_chunk(raw[line_end:next_p]))

    return "".join(pieces)


# ─────────────────────────────────────────────
# 2. PARSE MỘT FILE LOG
# ─────────────────────────────────────────────
def parse_log_file(path: str, meta: dict) -> list:
    content = smart_read_log(path)

    markers = list(SPLIT_RE.finditer(content))
    rows = []

    for i, m in enumerate(markers):
        split_seed = int(m.group(1))
        header_extra = m.group(2)
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(content)
        block = content[start:end]

        if "SKIPPED" in header_extra:
            rows.append({**meta, "split_seed": split_seed, "status": "skipped"})
            continue

        final_idx = block.find("FINAL ENSEMBLE RESULT")
        if final_idx == -1:
            # Chạy chưa xong hoặc crash giữa đường — không tìm thấy block kết quả
            rows.append({**meta, "split_seed": split_seed, "status": "incomplete"})
            continue

        final_block = block[final_idx:]
        acc_m = re.search(r"Accuracy\s*:\s*([0-9.]+)", final_block)
        f1_m  = re.search(r"Macro-F1\s*:\s*([0-9.]+)", final_block)
        wf1_m = re.search(r"Weighted-F1\s*:\s*([0-9.]+)", final_block)

        if not (acc_m and f1_m and wf1_m):
            rows.append({**meta, "split_seed": split_seed, "status": "parse_error"})
            continue

        low_n_warning = "may be misleading" in final_block.lower() or \
                        "misleading" in final_block.lower()

        rows.append({
            **meta,
            "split_seed":   split_seed,
            "status":       "ok",
            "accuracy":     float(acc_m.group(1)),
            "macro_f1":     float(f1_m.group(1)),
            "weighted_f1":  float(wf1_m.group(1)),
            "low_n_warn":   low_n_warning,
        })

    return rows


# ─────────────────────────────────────────────
# 3. WALK ALL LOG FILES
# ─────────────────────────────────────────────
log_paths = sorted(glob.glob(os.path.join(args.log_dir, "*.log")))
log_paths = [p for p in log_paths if not os.path.basename(p).startswith("prepare_")]

if not log_paths:
    raise SystemExit(f"Không tìm thấy file .log nào trong '{args.log_dir}'. "
                      f"Kiểm tra lại --log_dir.")

all_rows = []
skipped_files = []

for path in log_paths:
    fname = os.path.basename(path)
    fm = FNAME_RE.match(fname)
    if not fm:
        skipped_files.append(fname)
        continue
    meta = {
        "cancer":      fm.group("cancer"),
        "conv_type":   fm.group("conv"),
        "fusion":      fm.group("fusion"),
        "lr":          fm.group("lr"),
        "mixup_alpha": fm.group("mixup").replace("_", "."),
    }
    all_rows.extend(parse_log_file(path, meta))

if skipped_files:
    print(f"⚠  Bỏ qua {len(skipped_files)} file không khớp định dạng tên: "
          f"{skipped_files[:5]}{' ...' if len(skipped_files) > 5 else ''}")

df = pd.DataFrame(all_rows)
if df.empty:
    raise SystemExit("Không trích được kết quả nào — kiểm tra log có đúng định dạng không.")

raw_csv = os.path.join(args.log_dir, "grid_results_raw.csv")
df.to_csv(raw_csv, index=False)
print(f"\n→ Đã lưu kết quả thô từng split: {raw_csv}")

n_skipped     = (df["status"] == "skipped").sum()
n_incomplete  = (df["status"].isin(["incomplete", "parse_error"])).sum()
n_ok          = (df["status"] == "ok").sum()
print(f"  Tổng số dòng: {len(df)} | ok={n_ok} | skipped(thiếu graph)={n_skipped} | "
      f"incomplete/parse_error={n_incomplete}")

if n_incomplete > 0:
    bad = df[df["status"].isin(["incomplete", "parse_error"])]
    print(f"  ⚠  {n_incomplete} run chưa hoàn tất hoặc lỗi parse — xem chi tiết trong "
          f"'{raw_csv}' (status != 'ok') trước khi tin tưởng bảng tổng kết.")


# ─────────────────────────────────────────────
# 4. AGGREGATE: mean ± std qua các split_seed, theo từng config
# ─────────────────────────────────────────────
ok_df = df[df["status"] == "ok"].copy()
if ok_df.empty:
    raise SystemExit("Không có run nào 'ok' để tổng hợp — tất cả đều skipped/incomplete.")

group_cols = ["cancer", "conv_type", "fusion", "lr", "mixup_alpha"]
summary = (
    ok_df.groupby(group_cols)
    .agg(
        n_splits     = ("split_seed", "count"),
        accuracy_mean    = ("accuracy", "mean"),
        accuracy_std     = ("accuracy", "std"),
        macro_f1_mean    = ("macro_f1", "mean"),
        macro_f1_std     = ("macro_f1", "std"),
        weighted_f1_mean = ("weighted_f1", "mean"),
        weighted_f1_std  = ("weighted_f1", "std"),
        any_low_n_warn   = ("low_n_warn", "any"),
    )
    .reset_index()
)
summary["macro_f1_std"]    = summary["macro_f1_std"].fillna(0.0)
summary["weighted_f1_std"] = summary["weighted_f1_std"].fillna(0.0)
summary["accuracy_std"]    = summary["accuracy_std"].fillna(0.0)

rank_col = args.metric + "_mean"
summary = summary.sort_values(rank_col, ascending=False).reset_index(drop=True)
summary.insert(0, "rank", summary.index + 1)

summary_csv = os.path.join(args.log_dir, "grid_results_summary.csv")
summary.to_csv(summary_csv, index=False)
print(f"→ Đã lưu bảng tổng hợp theo config: {summary_csv}\n")


# ─────────────────────────────────────────────
# 5. PRINT TOP 10
# ─────────────────────────────────────────────
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print("=" * 100)
print(f"TOP 10 CONFIG (xếp hạng theo {args.metric}, qua {ok_df['split_seed'].nunique()} split_seed)")
print("=" * 100)
display_cols = ["rank", "conv_type", "fusion", "lr", "mixup_alpha", "n_splits",
                 "macro_f1_mean", "macro_f1_std", "weighted_f1_mean", "accuracy_mean"]
print(summary[display_cols].head(10).to_string(index=False))

incomplete_configs = summary[summary["n_splits"] < ok_df["split_seed"].nunique()]
if not incomplete_configs.empty:
    print(f"\n⚠  {len(incomplete_configs)} config chỉ có <5 split hoàn tất — "
          f"mean/std của các config này kém tin cậy hơn, xem cột n_splits.")

low_n_configs = summary[summary["any_low_n_warn"]]
if not low_n_configs.empty:
    print(f"⚠  {len(low_n_configs)} config có ít nhất 1 split mà macro-F1 bị cảnh báo "
          f"'misleading' (lớp test ≤2 mẫu) — nên đối chiếu thêm weighted_f1_mean.")


# ─────────────────────────────────────────────
# 6. BEST CONFIG + LỆNH REPRODUCE
# ─────────────────────────────────────────────
best = summary.iloc[0]
print("\n" + "=" * 100)
print(f"BEST CONFIG (theo {args.metric}):")
print("=" * 100)
print(f"  conv_type    : {best['conv_type']}")
print(f"  fusion       : {best['fusion']}")
print(f"  lr           : {best['lr']}")
print(f"  mixup_alpha  : {best['mixup_alpha']}")
print(f"  Macro-F1     : {best['macro_f1_mean']:.4f} ± {best['macro_f1_std']:.4f}  "
      f"(n={int(best['n_splits'])} splits)")
print(f"  Weighted-F1  : {best['weighted_f1_mean']:.4f} ± {best['weighted_f1_std']:.4f}")
print(f"  Accuracy     : {best['accuracy_mean']:.4f} ± {best['accuracy_std']:.4f}")

print(f"\n  Lệnh để chạy lại 1 split bất kỳ với config này:")
print(f'  python main.py --cancer {best["cancer"]} --data_path "E:\\Cancer-classification-dataset" '
      f'--split_seed 0 --conv_type {best["conv_type"]} --fusion {best["fusion"]} '
      f'--lr {best["lr"]} --mixup_alpha {best["mixup_alpha"]} --epochs 200 --patience 40')

print(f"\n  Để áp dụng config này cho 4 cohort còn lại (COAD/GBM/LGG/OV) theo giao thức 5×5:")
print(f'  foreach ($SEED in 0..4) {{')
print(f'      python scripts\\prepare_graph.py --cancer COAD --split_seed $SEED')
print(f'      python main.py --cancer COAD --data_path "E:\\Cancer-classification-dataset" '
      f'--split_seed $SEED --conv_type {best["conv_type"]} --fusion {best["fusion"]} '
      f'--lr {best["lr"]} --mixup_alpha {best["mixup_alpha"]} --epochs 200 --patience 40')
print(f'  }}  # lặp lại cho GBM, LGG, OV')


# ─────────────────────────────────────────────
# 7. CHART: Macro-F1 mean ± std theo config, tô màu theo (conv, fusion)
# ─────────────────────────────────────────────
plt.figure(figsize=(12, max(5, 0.35 * len(summary))))

color_map = {
    ("gat", "attn"):  "#378ADD",
    ("gat", "cross"): "#7B5CD6",
    ("gcn", "attn"):  "#EF9F27",
    ("gcn", "cross"): "#E24B4A",
}
labels = [f"{r.conv_type}/{r.fusion}  lr={r.lr}  mix={r.mixup_alpha}"
          for r in summary.itertuples()]
colors = [color_map.get((r.conv_type, r.fusion), "#999999") for r in summary.itertuples()]
y_pos  = np.arange(len(summary))[::-1]   # rank 1 ở trên cùng

plt.barh(y_pos, summary[rank_col], xerr=summary[args.metric + "_std"],
          color=colors, capsize=3, height=0.7)
plt.yticks(y_pos, labels, fontsize=8)
plt.xlabel(f"{args.metric} (mean ± std qua {ok_df['split_seed'].nunique()} split_seed)")
plt.title(f"{best['cancer']} — Grid search: {args.metric} theo config")

import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=c, label=f"{k[0]}/{k[1]}") for k, c in color_map.items()]
plt.legend(handles=handles, loc="lower right", fontsize=8)
plt.tight_layout()

chart_path = os.path.join(args.log_dir, "grid_results_chart.png")
plt.savefig(chart_path, dpi=130)
plt.close()
print(f"\n→ Đã lưu chart: {chart_path}")