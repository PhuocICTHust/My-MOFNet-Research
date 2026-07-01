import argparse
import os
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from models import PanCancerModel, FocalLoss


# ─────────────────────────────
# ARGS
# ─────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cancer",         required=True,
                    choices=["BRCA", "COAD", "GBM", "LGG", "OV"])
parser.add_argument("--data_path",      required=True)
# FIX P3 (loader side): load the split-tagged graph file.
parser.add_argument("--split_seed",     type=int,   default=777,
                    help="Identifies which graph file to load: "
                         "{cancer}_graph_s{split_seed}.pt. Vary across runs "
                         "(e.g. 0..4) for repeated stratified CV; report mean ± std "
                         "of the per-split test scores.")
parser.add_argument("--epochs",         type=int,   default=200,
                    help="Max epochs — early stopping handles actual termination.")
parser.add_argument("--nhid",           type=int,   default=64)
# PATCH 1: GAT-vs-GCN ablation switch. Run both, report both columns.
parser.add_argument("--conv_type",      choices=["gat", "gcn"], default="gat",
                    help="Graph encoder. 'gat'=attention (default); "
                         "'gcn'=lighter, ~50%% fewer params (often better on the "
                         "smallest cohorts GBM/OV/COAD). Run both for the ablation.")
# PATCH 2: fusion mechanism. 'attn'=soft attention (≈ paper's MVA ablation);
# 'cross'=VCDN-lite pairwise label-space fusion (targets the MVA→full-MOFNet gap).
parser.add_argument("--fusion",         choices=["attn", "cross"], default="attn",
                    help="Omics fusion. 'attn'=soft attention weighted-sum "
                         "(default, stable baseline); 'cross'=lightweight "
                         "cross-omics label fusion (validate per-cohort; can "
                         "overfit COAD/GBM).")
parser.add_argument("--num_classes",    type=int,   default=0)
parser.add_argument("--dropout_ratio",  type=float, default=0.3)
parser.add_argument("--lr",             type=float, default=5e-4)
parser.add_argument("--weight_decay",   type=float, default=5e-4)
parser.add_argument("--focal_gamma",    type=float, default=2.0)
parser.add_argument("--entropy_weight", type=float, default=0.01)
# Registered so getattr(args,"attention_temp") is actually configurable
# (previously dead config: temperature was always 1.0).
parser.add_argument("--attention_temp", type=float, default=1.0,
                    help="Softmax temperature for the modality fusion gate.")

# ── FIX A: Linear warmup before cosine decay ─────────────────────────────
parser.add_argument("--warmup_epochs",  type=int,   default=10,
                    help="Linear LR warmup epochs before cosine decay. 0 = disabled.")

# ── FIX B: Early stopping ────────────────────────────────────────────────
parser.add_argument("--patience",       type=int,   default=40,
                    help="Early stopping patience on val F1. 0 = disabled.")

# ── FIX C: Label smoothing ───────────────────────────────────────────────
parser.add_argument("--label_smoothing", type=float, default=0.05,
                    help="Label smoothing ε for FocalLoss. 0 = disabled.")

# ── FIX D: Best-checkpoint ensemble ──────────────────────────────────────
# Now ALWAYS on. The previous last-10-epoch fallback required scoring the test
# set every epoch, which invites researcher-degrees-of-freedom leakage. Test
# is now touched exactly once: at the end, from the best-val-F1 checkpoints.
parser.add_argument("--use_best_ensemble", action="store_true",
                    help="(Retained for CLI compatibility; ensemble is always "
                         "built from best-val-F1 checkpoints now.)")

# ── FIX E: Feature MixUp ────────────────────────────────────────────────
# Mixed loss: λ·L(f(x̃), yₐ) + (1-λ)·L(f(x̃), y_b), λ ~ Beta(α,α).
# NOTE (FIX A2): the cross-label term is now scored against the SAME mixed
# predictions (see train()). The original code indexed logits[perm], which made
# focal_b numerically identical to focal_a → MixUp collapsed to a no-op.
parser.add_argument("--mixup_alpha",    type=float, default=0.2,
                    help="MixUp Beta parameter for feature interpolation. 0 = disabled. "
                         "Consider 0.1 to protect tiny minority classes.")

args = parser.parse_args()


# ─────────────────────────────
# SEED  (FIX A4: determinism)
# ─────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_graph():
    # FIX P3: split-tagged filename, with backward-compatible fallback.
    tagged = os.path.join(args.data_path, f"{args.cancer}_graph_s{args.split_seed}.pt")
    legacy = os.path.join(args.data_path, f"{args.cancer}_graph.pt")
    path = tagged if os.path.exists(tagged) else legacy

    # FIX (diagnostics): the previous version let torch.load() raise its bare
    # FileNotFoundError on the legacy path, which looks identical whether the
    # cause was "wrong split_seed", "prepare_graph.py never ran", or
    # "prepare_graph.py crashed silently upstream". State both paths that were
    # tried so the next failure is diagnosable in one glance, no detective work.
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No graph file found for cancer={args.cancer}, split_seed={args.split_seed}.\n"
            f"  Tried (tagged): {tagged}\n"
            f"  Tried (legacy): {legacy}\n"
            f"  -> Run: python scripts/prepare_graph.py --cancer {args.cancer} "
            f"--split_seed {args.split_seed}\n"
            f"     and confirm it actually prints '... DONE! Saved to: ...' "
            f"(check exit code / log if run from a script)."
        )
    return torch.load(path, weights_only=False)


# ─────────────────────────────
# TRAIN  (with optional MixUp)
# ─────────────────────────────
def train(model, data, optimizer, criterion, mask):
    model.train()
    d1, d2, d3, d4 = data['mRNA'], data['miRNA'], data['Methy'], data['CNV']
    optimizer.zero_grad()

    device   = next(model.parameters()).device
    mask_dev = mask.to(device)
    y_dev    = d1.y.to(device)

    if args.mixup_alpha > 0:
        # ── Feature-space MixUp on training nodes only ───────────────────
        # Interpolate features of randomly-paired training nodes. Graph
        # topology is unchanged; only node features are mixed. Loss is the
        # convex combination of the two constituent class losses.
        train_idx = mask_dev.nonzero(as_tuple=True)[0]
        n         = len(train_idx)
        perm      = train_idx[torch.randperm(n, device=device)]
        lam       = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))

        # Save originals and mix in-place (restore after forward)
        saved = {}
        for key, d in [('mRNA', d1), ('miRNA', d2), ('Methy', d3), ('CNV', d4)]:
            saved[key] = d.x[train_idx].clone()
            d.x        = d.x.clone()
            d.x[train_idx] = lam * d.x[train_idx] + (1 - lam) * d.x[perm]

        logits, entropy = model(d1, d2, d3, d4)

        for key, d in [('mRNA', d1), ('miRNA', d2), ('Methy', d3), ('CNV', d4)]:
            d.x[train_idx] = saved[key]

        # FIX A2: both terms use the SAME mixed predictions logits[train_idx];
        # only the labels differ (yₐ vs y_b). The original logits[perm] made
        # focal_b ≡ focal_a, nullifying the cross-label MixUp signal.
        focal_a    = criterion(logits[train_idx], y_dev[train_idx])
        focal_b    = criterion(logits[train_idx], y_dev[perm])
        focal_loss = lam * focal_a + (1 - lam) * focal_b
    else:
        logits, entropy = model(d1, d2, d3, d4)
        focal_loss = criterion(logits[mask_dev], y_dev[mask_dev])

    loss = focal_loss - args.entropy_weight * entropy
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()
    return loss.item(), focal_loss.item()


# ─────────────────────────────
# EVALUATE
# ─────────────────────────────
def evaluate(model, data, mask):
    model.eval()
    d1, d2, d3, d4 = data['mRNA'], data['miRNA'], data['Methy'], data['CNV']
    with torch.no_grad():
        logits, _ = model(d1, d2, d3, d4)
        pred      = logits.argmax(dim=1)
    y    = d1.y.cpu()
    pred = pred.cpu()
    mask = mask.cpu()
    acc  = accuracy_score(y[mask], pred[mask])
    f1   = f1_score(y[mask], pred[mask], average="macro", zero_division=0)
    attn = model.get_attention_weights().cpu()
    return acc, f1, logits.cpu(), attn


# ─────────────────────────────
# RUN ONE SEED  (model-init variance; data split is fixed by split_seed)
# ─────────────────────────────
def run_one_seed(seed):
    print(f"\n{'='*50}")
    print(f"  Init seed {seed}  (split_seed={args.split_seed})")
    print(f"{'='*50}")
    set_seed(seed)

    data   = load_graph()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    for k in data:
        data[k] = data[k].to(device)

    model = PanCancerModel(args, data).to(device)

    # ── FIX A1: class weights from TRAIN labels only (no val/test leakage) ──
    y_full        = data["mRNA"].y.cpu().numpy()
    train_mask_np = data["mRNA"].train_mask.cpu().numpy()
    y_train       = y_full[train_mask_np]
    classes       = np.unique(y_full)            # full set → stable weight length
    present       = np.unique(y_train)
    w_present     = compute_class_weight("balanced", classes=present, y=y_train)
    w             = np.ones(len(classes), dtype=np.float64)
    for c, wc in zip(present, w_present):
        w[c] = wc
    w   = np.clip(w, 0.5, 5.0)
    w_t = torch.tensor(w, dtype=torch.float32).to(device)
    print(f"  Class weights (train-only, clamped): {np.round(w, 4)}")

    criterion = FocalLoss(gamma=args.focal_gamma, alpha=w_t,
                          label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)

    # ── FIX A: Warmup + cosine scheduler ─────────────────────────────────
    if args.warmup_epochs > 0:
        n_cosine = max(1, args.epochs - args.warmup_epochs)
        warmup_sched = LinearLR(optimizer, start_factor=0.1,
                                end_factor=1.0, total_iters=args.warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=n_cosine, eta_min=1e-6)
        scheduler = SequentialLR(optimizer,
                                 schedulers=[warmup_sched, cosine_sched],
                                 milestones=[args.warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_mask = data["mRNA"].train_mask.to(device)
    val_mask   = data["mRNA"].val_mask.to(device)

    best_val_f1    = 0.0
    patience_ctr   = 0
    best_epoch     = 0
    val_f1_history = []

    ckpt_dir  = os.path.join(args.data_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Tag with conv_type + fusion so GAT/GCN × attn/cross runs never collide.
    ckpt_path = os.path.join(
        ckpt_dir,
        f"{args.cancer}_{args.conv_type}_{args.fusion}_s{args.split_seed}_seed{seed}_best.pth")

    neg_loss_warned = False

    for epoch in range(args.epochs):
        loss, focal = train(model, data, optimizer, criterion, train_mask)
        scheduler.step()

        # Total loss can go negative when entropy is near maximum (log(4)≈1.39)
        # and focal is tiny. NOTE: this is also a sign that TRAIN focal has been
        # driven near zero (focal < entropy_weight·entropy ⇒ focal < ~0.014),
        # i.e. an overfitting indicator in this small-N regime — not just a
        # benign artefact. Selection is on val F1, so it does not corrupt
        # model choice, but treat persistent negative loss as a cue to add
        # regularisation rather than to ignore.
        if loss < 0 and not neg_loss_warned:
            print(f"  [ep {epoch}] Total loss negative (focal={focal:.4f}): "
                  f"entropy near max / train focal near zero — watch for overfitting.")
            neg_loss_warned = True

        # FIX A3: select on VAL only. Test is NOT scored inside the loop, to
        # avoid surfacing test F1 every epoch (a researcher-leakage channel).
        # FIX P4: evaluate() already computes model.get_attention_weights()
        # internally (a full-graph forward pass, so it covers ALL N patients,
        # not just val) — it was just being discarded with "_". Capture it so
        # the checkpoint below can save it for interpretability plots.
        val_acc, val_f1, _, val_attn = evaluate(model, data, val_mask)
        val_f1_history.append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_epoch   = epoch
            patience_ctr = 0
            torch.save({
                'epoch':             epoch,
                'model_state_dict':  model.state_dict(),
                'val_f1':            val_f1,
                'args':              vars(args),
                'y':                 data['mRNA'].y.cpu(),
                'omics_names':       ['mRNA', 'miRNA', 'Methy', 'CNV'],
                # FIX P4: per-patient [N,4] weights for the interpretability
                # plot. CAUTION when args.fusion == "cross": CrossOmicsFusion
                # does not have real attention — this is a mean per-omic head
                # confidence proxy (see models.py docstring), not a learned
                # fusion weight. visualize_attention.py reads args['fusion']
                # from this same checkpoint to label the chart correctly.
                'attention_weights': val_attn,
            }, ckpt_path)
        else:
            patience_ctr += 1

        # ── FIX B: Early stopping ─────────────────────────────────────────
        if args.patience > 0 and patience_ctr >= args.patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(best val F1={best_val_f1:.4f} @ ep {best_epoch})")
            break

        if epoch % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Ep {epoch:3d} | loss={loss:.4f} focal={focal:.4f} | "
                  f"val_acc={val_acc:.4f} val_F1={val_f1:.4f} | lr={lr_now:.2e}")

    print(f"  → Best val macro-F1 (init seed {seed}): {best_val_f1:.4f} @ epoch {best_epoch}")
    print(f"  → Checkpoint: {ckpt_path}")

    # ── Val F1 curve ──────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(val_f1_history, linewidth=1.5, color='steelblue', label='Val Macro-F1')
    plt.axvline(x=best_epoch, color='tomato', linestyle='--', linewidth=1,
                label=f'Best epoch {best_epoch} (val F1={best_val_f1:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('Val Macro-F1')
    plt.title(f'{args.cancer} (split {args.split_seed}) — init seed {seed}')
    plt.legend(fontsize=9)
    plt.tight_layout()
    plot_path = os.path.join(args.data_path,
                             f"f1_curve_{args.cancer}_{args.conv_type}_{args.fusion}_s{args.split_seed}_seed{seed}.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  → Val F1 curve: {plot_path}")

    return data, val_f1_history


# ─────────────────────────────
# MAIN
# ─────────────────────────────
if __name__ == "__main__":

    if args.num_classes == 0:
        _tmp = load_graph()
        args.num_classes = int(_tmp["mRNA"].y.max().item()) + 1
        del _tmp

    print(f"\n  Cancer      : {args.cancer}")
    print(f"  split_seed  : {args.split_seed}  (data split fixed by this)")
    print(f"  num_classes : {args.num_classes}  (BRCA/GBM=5, COAD/OV=4, LGG=3)")
    print(f"  warmup      : {args.warmup_epochs} epochs")
    print(f"  patience    : {args.patience}")
    print(f"  label_smth  : {args.label_smoothing}")
    print(f"  mixup_alpha : {args.mixup_alpha}")
    print(f"  conv_type   : {args.conv_type}")
    print(f"  fusion      : {args.fusion}")
    print(f"  ensemble    : best-checkpoint (always)")

    # These vary weight init / dropout / mixup RNG only — the DATA SPLIT is
    # fixed by --split_seed. For honest small-N variance, sweep --split_seed
    # (e.g. 0..4) across separate runs and aggregate the per-split test scores.
    seeds         = [777, 42, 1234, 2024, 999]
    all_f1_curves = []
    final_data    = None
    final_device  = None

    for s in seeds:
        data, f1_hist = run_one_seed(s)
        all_f1_curves.append(f1_hist)
        final_data   = data
        final_device = next(iter(data.values())).x.device

    # ── Summary val F1 plot ───────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    colors = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple', 'crimson']
    for idx, (hist, seed) in enumerate(zip(all_f1_curves, seeds)):
        plt.plot(hist, linewidth=1.2, alpha=0.8,
                 color=colors[idx], label=f'Init seed {seed}')
    max_len = max(len(c) for c in all_f1_curves)

    arr = np.full((len(all_f1_curves), max_len), np.nan)
    for i, curve in enumerate(all_f1_curves):
        arr[i, :len(curve)] = curve

    mean_curve = np.nanmean(arr, axis=0)
    plt.plot(mean_curve, linewidth=2.5, color='black',
             linestyle='--', label='Mean')
    plt.xlabel('Epoch')
    plt.ylabel('Val Macro-F1')
    plt.title(f'{args.cancer} (split {args.split_seed}) — all init seeds')
    plt.legend(fontsize=9)
    plt.tight_layout()
    summary_plot = os.path.join(args.data_path,
                                f"f1_curve_{args.cancer}_{args.conv_type}_{args.fusion}_s{args.split_seed}_all_seeds.png")
    plt.savefig(summary_plot, dpi=120)
    plt.close()
    print(f"\n  → Summary val F1 curve: {summary_plot}")

    # ── FIX D + A3: Best-checkpoint ensemble (test touched exactly once) ──
    ckpt_dir = os.path.join(args.data_path, "checkpoints")
    device   = final_device

    print("\n  Loading best-val checkpoints for ensemble...")
    best_ckpt_logits = []
    for seed in seeds:
        ckpt_path = os.path.join(
            ckpt_dir,
            f"{args.cancer}_{args.conv_type}_{args.fusion}_s{args.split_seed}_seed{seed}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"  ⚠  Checkpoint not found for init seed {seed}, skipping.")
            continue
        ckpt = torch.load(ckpt_path, weights_only=False)
        model_tmp = PanCancerModel(args, final_data).to(device)
        model_tmp.load_state_dict(ckpt['model_state_dict'])
        _, _, logits_cpu, _ = evaluate(model_tmp, final_data,
                                       final_data["mRNA"].test_mask)
        best_ckpt_logits.append(logits_cpu)
        print(f"    Init seed {seed}: best ep={ckpt.get('epoch','?')}, "
              f"val_F1={ckpt.get('val_f1', float('nan')):.4f}")

    if not best_ckpt_logits:
        raise RuntimeError("No checkpoints found — cannot build ensemble.")

    ensemble = torch.stack(best_ckpt_logits).mean(dim=0)
    pred     = ensemble.argmax(dim=1)

    y    = final_data["mRNA"].y.cpu()
    mask = final_data["mRNA"].test_mask.cpu()

    acc         = accuracy_score(y[mask], pred[mask])
    y_arr       = y[mask].numpy()
    pred_arr    = pred[mask].numpy()
    f1_macro    = f1_score(y_arr, pred_arr, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_arr, pred_arr, average="weighted", zero_division=0)

    counts_test = np.bincount(y_arr, minlength=args.num_classes)
    min_tc      = int(counts_test.min())

    print("\n" + "=" * 57)
    print("  FINAL ENSEMBLE RESULT (test set — never seen during training)")
    print(f"  (split_seed={args.split_seed})")
    print("=" * 57)
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro-F1    : {f1_macro:.4f}")

    # When a test class has ≤2 samples, macro-F1 is dominated by a coin flip on
    # one or two points; report weighted-F1 as the fair comparison metric.
    if min_tc < 3:
        print(f"  Weighted-F1 : {f1_weighted:.4f}  "
              f"<-- preferred metric (min test-class count = {min_tc})")
        print(f"  ⚠  Class(es) with ≤2 test samples make macro-F1 misleading. "
              f"Report weighted-F1 for fair comparison.")
    else:
        print(f"  Weighted-F1 : {f1_weighted:.4f}")

    y_all     = y.numpy()
    counts    = np.bincount(y_all, minlength=args.num_classes)
    cls_names = [f"Class{i}({counts[i]})" for i in range(args.num_classes)]

    print("  Per-class report:")
    print(classification_report(y_arr, pred_arr,
                                target_names=cls_names, zero_division=0))

    csv_path = os.path.join(args.data_path,
                            f"{args.cancer}_{args.conv_type}_{args.fusion}_s{args.split_seed}_ensemble_predictions.csv")
    pd.DataFrame({
        'true_label': y_arr,
        'pred_label': pred_arr,
    }).to_csv(csv_path, index=False)
    print(f"\n  Predictions saved: {csv_path}")
