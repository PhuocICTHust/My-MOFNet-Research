# ==========================================
# RUN TOP-3 CONFIGS (1 lần / config) — lấy checkpoint cho interpretability
# ==========================================
# Mục đích: KHÔNG chạy lại grid search hay 5x5 protocol. Chỉ chạy main.py
# đúng 1 lần (ở split_seed=0, khớp với checkpoint bạn đã có) cho mỗi config
# trong top-3 từ grid_results_summary.csv. Mỗi lần chạy vẫn train nội bộ
# 5 init seed (thiết kế gốc của main.py) → cho ra 5 checkpoint .pth/config.
# Cần main.py đã được FIX P4 (lưu attention_weights) trước khi chạy script này.

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING     = "utf-8"
$env:PYTHONUTF8           = "1"

$DATA_DIR    = "E:\Cancer-classification-dataset"
$LOG_DIR     = ".\logs_topconfigs"
$PREP_SCRIPT = "scripts\prepare_graph.py"
$MAIN_SCRIPT = "main.py"
$CANCER      = "BRCA"
$SPLIT_SEED  = 0          # khớp với BRCA_..._s0_... bạn đã có sẵn

if (-Not (Test-Path -Path $LOG_DIR)) {
    New-Item -ItemType Directory -Path $LOG_DIR | Out-Null
}
foreach ($script in @($PREP_SCRIPT, $MAIN_SCRIPT)) {
    if (-Not (Test-Path -Path $script)) {
        Write-Host "FATAL: cannot find '$script' relative to $(Get-Location)." -ForegroundColor Red
        exit 1
    }
}

# Top-3 config TRONG NHOM fusion=attn (theo grid_results_summary.csv, BRCA).
# fusion=cross KHONG duoc dua vao day: CrossOmicsFusion khong co attention
# thuc su (chi la mean per-omic head confidence - xem models.py docstring),
# nen khong dung de chay visualize_attention.py duoc.
$CONFIGS = @(
    @{ Conv = "gat"; Fusion = "attn"; LR = "5e-4"; Mixup = "0.0" },  # #1/attn: macro_f1=0.7651, std=0.0159 (thap nhat)
    @{ Conv = "gcn"; Fusion = "attn"; LR = "1e-3"; Mixup = "0.0" },  # #2/attn: macro_f1=0.7642
    @{ Conv = "gat"; Fusion = "attn"; LR = "1e-3"; Mixup = "0.2" }   # #3/attn: macro_f1=0.7639
)

# Đảm bảo graph file split_seed=0 đã tồn tại — build nếu chưa có.
$graphFile = Join-Path $DATA_DIR "${CANCER}_graph_s${SPLIT_SEED}.pt"
if (-Not (Test-Path -Path $graphFile)) {
    Write-Host ">> Graph file chưa có, build 1 lần..." -ForegroundColor Magenta
    $prepLog = "$LOG_DIR\prepare_${CANCER}_s${SPLIT_SEED}.log"
    & python $PREP_SCRIPT --cancer $CANCER --split_seed $SPLIT_SEED *> $prepLog
    if ($LASTEXITCODE -ne 0 -or -Not (Test-Path -Path $graphFile)) {
        Write-Host "FATAL: prepare_graph.py thất bại. Xem $prepLog" -ForegroundColor Red
        exit 1
    }
}

$failures = @()

foreach ($cfg in $CONFIGS) {
    $CONV   = $cfg.Conv
    $FUSION = $cfg.Fusion
    $LR     = $cfg.LR
    $MIXUP  = $cfg.Mixup
    $LOG_FILE = "$LOG_DIR\${CANCER}_${CONV}_${FUSION}_lr${LR}_mixup${MIXUP}_s${SPLIT_SEED}.log"

    Write-Host ">> Training: $CANCER | conv=$CONV | fusion=$FUSION | lr=$LR | mixup=$MIXUP (split_seed=$SPLIT_SEED)" -ForegroundColor Yellow

    $mainArgs = @(
        "--cancer", $CANCER,
        "--data_path", $DATA_DIR,
        "--split_seed", $SPLIT_SEED,
        "--conv_type", $CONV,
        "--fusion", $FUSION,
        "--lr", $LR,
        "--mixup_alpha", $MIXUP,
        "--epochs", "200",
        "--patience", "40"
    )
    & python $MAIN_SCRIPT @mainArgs *> $LOG_FILE
    $exit = $LASTEXITCODE

    if ($exit -ne 0) {
        Write-Host "  !! FAILED (exit=$exit). Xem $LOG_FILE" -ForegroundColor Red
        $failures += "$CONV/$FUSION/lr$LR/mix$MIXUP | exit=$exit | log=$LOG_FILE"
    } else {
        Write-Host "  OK -> checkpoints\${CANCER}_${CONV}_${FUSION}_s${SPLIT_SEED}_seed*_best.pth (5 file)" -ForegroundColor Green
    }
}

Write-Host ""
if ($failures.Count -eq 0) {
    Write-Host "DONE -- ca 3 config da train xong, checkpoint da co attention_weights." -ForegroundColor Cyan
} else {
    Write-Host "DONE VOI LOI:" -ForegroundColor Red
    $failures | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
}
