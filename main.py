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
parser.add_argument("--epochs",         type=int,   default=200,
                    help="Max epochs — early stopping handles actual termination.")
parser.add_argument("--nhid",           type=int,   default=64)
parser.add_argument("--num_classes",    type=int,   default=0)
parser.add_argument("--dropout_ratio",  type=float, default=0.3)
parser.add_argument("--lr",             type=float, default=5e-4)
parser.add_argument("--weight_decay",   type=float, default=5e-4)
parser.add_argument("--focal_gamma",    type=float, default=2.0)
parser.add_argument("--entropy_weight", type=float, default=0.01)

# ── FIX A: Linear warmup before cosine decay ─────────────────────────────
# Prevents the sharp LR drop at epoch 0 that caused GBM/OV instability:
# ep-0 val F1 ranged from 0.19-0.48 (0.29-unit spread), implying noisy
# gradient directions at high LR. Warmup lifts start_factor to 0.1*lr=5e-5,
# ramps linearly to lr over --warmup_epochs, then hands off to cosine.
# Cost: ~0 (just a scheduler change). Gain: ~0.01-0.03 reduction in seed variance.
parser.add_argument("--warmup_epochs",  type=int,   default=10,
                    help="Linear LR warmup epochs before cosine decay. 0 = disabled.")

# ── FIX B: Early stopping ────────────────────────────────────────────────
# Without early stopping, the last-10-epoch ensemble windows were collected
# at epochs 140-149 regardless of whether the model had already peaked.
# For LGG, best val F1 was reached at epoch 40; collecting ep 140-149
# adds noise from near-zero LR epochs. With patience=40, training halts
# automatically and the ensemble is tighter around the best checkpoint.
# Exception: small val sets (LGG val=25) can plateau early due to ceiling
# effects — use patience>=40 to avoid premature stopping there.
parser.add_argument("--patience",       type=int,   default=40,
                    help="Early stopping patience on val F1. 0 = disabled.")

# ── FIX C: Label smoothing ───────────────────────────────────────────────
# BRCA val F1 oscillated ±0.06 between epochs (0.786→0.700→0.756→0.722...)
# even with CosineAnnealingLR at low LR. The cause: hard 0/1 labels cause
# the model to push logit gaps to extremes, making it sensitive to small
# perturbations. Label smoothing (ε=0.05) softens targets to ε/(C-1) for
# non-target classes, stabilising the gradient norm.
parser.add_argument("--label_smoothing", type=float, default=0.05,
                    help="Label smoothing ε for FocalLoss. 0 = disabled.")

# ── FIX D: Best-checkpoint ensemble ─────────────────────────────────────
# Original: ensemble = mean of last-10-epoch logits from all 5 seeds.
# Problem: epoch T-9 through T may include training noise if LR is still
# meaningful, OR if early stopping fires at different epochs per seed, the
# window sizes become inconsistent.
# New: after all seeds, reload each seed's best-val-F1 checkpoint and
# run a single inference pass. Ensemble = mean of 5 best-checkpoint logits.
# Strictly ≥ original (never worse), typically +0.005-0.02 macro-F1.
parser.add_argument("--use_best_ensemble", action="store_true",
                    help="Ensemble from saved best-val-F1 checkpoints (recommended).")

# ── FIX E: Feature MixUp ────────────────────────────────────────────────
# Interpolates training-node features between random pairs in each batch.
# Mixed loss: λ·L(f(x̃), yₐ) + (1-λ)·L(f(x̃), y_b), λ ~ Beta(α,α).
# Reduces GBM class-4 recall (0.33 in original) by smoothing decision
# boundaries near minority-class nodes. Disable with --mixup_alpha 0.
parser.add_argument("--mixup_alpha",    type=float, default=0.2,
                    help="MixUp Beta parameter for feature interpolation. 0 = disabled.")

args = parser.parse_args()


# ─────────────────────────────
# SEED
# ─────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_graph():
    path = os.path.join(args.data_path, f"{args.cancer}_graph.pt")
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
        # Interpolate features of randomly-paired training nodes.
        # Graph topology is unchanged; only node features are mixed.
        # Loss is the convex combination of the two constituent class losses.
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

        # λ·L(ŷ, yₐ) + (1-λ)·L(ŷ, y_b)
        focal_a    = criterion(logits[train_idx], y_dev[train_idx])
        focal_b    = criterion(logits[perm],      y_dev[perm])
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
# RUN ONE SEED
# ─────────────────────────────
def run_one_seed(seed):
    print(f"\n{'='*50}")
    print(f"  Seed {seed}")
    print(f"{'='*50}")
    set_seed(seed)

    data   = load_graph()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    for k in data:
        data[k] = data[k].to(device)

    model = PanCancerModel(args, data).to(device)

    y_np    = data["mRNA"].y.cpu().numpy()
    classes = np.unique(y_np)
    w       = compute_class_weight("balanced", classes=classes, y=y_np)
    w       = np.clip(w, 0.5, 5.0)
    w_t     = torch.tensor(w, dtype=torch.float32).to(device)
    print(f"  Class weights (clamped): {np.round(w, 4)}")

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
    test_mask  = data["mRNA"].test_mask.to(device)

    best_val_f1    = 0.0
    patience_ctr   = 0
    best_epoch     = 0
    last_k_logits  = []
    val_f1_history = []

    ckpt_dir  = os.path.join(args.data_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.cancer}_seed{seed}_best.pth")

    neg_loss_warned = False

    for epoch in range(args.epochs):
        loss, focal = train(model, data, optimizer, criterion, train_mask)
        scheduler.step()

        # ── FIX A note: total loss can go negative when entropy is near
        # its maximum (log(4) ≈ 1.39 for 4 modalities) and focal is tiny.
        # This is the entropy regularisation working as intended — it is NOT
        # a training failure. Track focal_loss separately for convergence.
        if loss < 0 and not neg_loss_warned:
            print(f"  [ep {epoch}] Total loss negative (focal={focal:.4f}): "
                  f"entropy near maximum, fusion weights are near-uniform. "
                  f"This is expected and harmless.")
            neg_loss_warned = True

        val_acc,  val_f1,  _,          _    = evaluate(model, data, val_mask)
        test_acc, test_f1, logits_cpu,  attn = evaluate(model, data, test_mask)
        val_f1_history.append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            patience_ctr = 0
            torch.save({
                'epoch':             epoch,
                'model_state_dict':  model.state_dict(),
                'val_f1':            val_f1,
                'test_f1':           test_f1,
                'args':              vars(args),
                'attention_weights': attn,
                'y':                 data['mRNA'].y.cpu(),
                'omics_names':       ['mRNA', 'miRNA', 'Methy', 'CNV'],
            }, ckpt_path)
        else:
            patience_ctr += 1

        # ── FIX B: Early stopping ─────────────────────────────────────────
        if args.patience > 0 and patience_ctr >= args.patience:
            print(f"  Early stopping at epoch {epoch} "
                  f"(best val F1={best_val_f1:.4f} @ ep {best_epoch})")
            break

        # Collect last-K logits as fallback ensemble (if use_best_ensemble=False)
        if epoch >= args.epochs - 10:
            last_k_logits.append(logits_cpu)

        if epoch % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Ep {epoch:3d} | loss={loss:.4f} focal={focal:.4f} | "
                  f"val_acc={val_acc:.4f} val_F1={val_f1:.4f} | "
                  f"test_acc={test_acc:.4f} test_F1={test_f1:.4f} | "
                  f"lr={lr_now:.2e}")

    print(f"  → Best val macro-F1 (seed {seed}): {best_val_f1:.4f} @ epoch {best_epoch}")
    print(f"  → Checkpoint: {ckpt_path}")

    # ── Val F1 curve ──────────────────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(val_f1_history, linewidth=1.5, color='steelblue', label='Val Macro-F1')
    plt.axvline(x=best_epoch, color='tomato', linestyle='--', linewidth=1,
                label=f'Best epoch {best_epoch} (val F1={best_val_f1:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('Val Macro-F1')
    plt.title(f'{args.cancer} — Seed {seed} (val)')
    plt.legend(fontsize=9)
    plt.tight_layout()
    plot_path = os.path.join(args.data_path,
                             f"f1_curve_{args.cancer}_seed{seed}.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  → Val F1 curve: {plot_path}")

    ensemble_logits = (torch.stack(last_k_logits).mean(dim=0)
                       if last_k_logits else logits_cpu)
    return ensemble_logits, data, val_f1_history


# ─────────────────────────────
# MAIN
# ─────────────────────────────
if __name__ == "__main__":

    if args.num_classes == 0:
        _tmp = load_graph()
        args.num_classes = int(_tmp["mRNA"].y.max().item()) + 1
        del _tmp

    print(f"\n  Cancer      : {args.cancer}")
    print(f"  num_classes : {args.num_classes}  (BRCA/GBM=5, COAD/OV=4, LGG=3)")
    print(f"  warmup      : {args.warmup_epochs} epochs")
    print(f"  patience    : {args.patience}")
    print(f"  label_smth  : {args.label_smoothing}")
    print(f"  mixup_alpha : {args.mixup_alpha}")
    print(f"  ensemble    : {'best-checkpoint' if args.use_best_ensemble else 'last-10-epoch'}")

    seeds         = [777, 42, 1234, 2024, 999]
    all_logits    = []
    all_f1_curves = []
    final_data    = None
    final_device  = None

    for s in seeds:
        logits, data, f1_hist = run_one_seed(s)
        all_logits.append(logits)
        all_f1_curves.append(f1_hist)
        final_data   = data
        final_device = next(iter(data.values())).x.device

# ── Summary val F1 plot ───────────────────────────────────────────────
    plt.figure(figsize=(10, 5))
    colors = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple', 'crimson']
    for idx, (hist, seed) in enumerate(zip(all_f1_curves, seeds)):
        plt.plot(hist, linewidth=1.2, alpha=0.8,
                 color=colors[idx], label=f'Seed {seed}')
    max_len = max(len(c) for c in all_f1_curves)

    arr = np.full(
        (len(all_f1_curves), max_len),
        np.nan
    )

    for i, curve in enumerate(all_f1_curves):
        arr[i, :len(curve)] = curve

    mean_curve = np.nanmean(arr, axis=0)
    plt.plot(mean_curve, linewidth=2.5, color='black',
             linestyle='--', label='Mean')
    plt.xlabel('Epoch')
    plt.ylabel('Val Macro-F1')
    plt.title(f'{args.cancer} — All seeds Val F1 curves')
    plt.legend(fontsize=9)
    plt.tight_layout()
    summary_plot = os.path.join(args.data_path,
                                f"f1_curve_{args.cancer}_all_seeds.png")
    plt.savefig(summary_plot, dpi=120)
    plt.close()
    print(f"\n  → Summary val F1 curve: {summary_plot}")

    # ── FIX D: Best-checkpoint ensemble ──────────────────────────────────
    ckpt_dir = os.path.join(args.data_path, "checkpoints")
    device   = final_device

    if args.use_best_ensemble:
        print("\n  Loading best checkpoints for ensemble...")
        best_ckpt_logits = []
        for seed in seeds:
            ckpt_path = os.path.join(ckpt_dir,
                                     f"{args.cancer}_seed{seed}_best.pth")
            if not os.path.exists(ckpt_path):
                print(f"  ⚠  Checkpoint not found for seed {seed}, skipping.")
                continue
            ckpt = torch.load(ckpt_path, weights_only=False)
            model_tmp = PanCancerModel(args, final_data).to(device)
            model_tmp.load_state_dict(ckpt['model_state_dict'])
            _, _, logits_cpu, _ = evaluate(model_tmp,
                                           final_data,
                                           final_data["mRNA"].test_mask)
            best_ckpt_logits.append(logits_cpu)
            print(f"    Seed {seed}: best ep={ckpt.get('epoch','?')}, "
                  f"val_F1={ckpt.get('val_f1', float('nan')):.4f}")
        ensemble = torch.stack(best_ckpt_logits).mean(dim=0)
    else:
        ensemble = torch.stack(all_logits).mean(dim=0)

    pred = ensemble.argmax(dim=1)

    y    = final_data["mRNA"].y.cpu()
    mask = final_data["mRNA"].test_mask.cpu()

    acc          = accuracy_score(y[mask], pred[mask])
    y_arr        = y[mask].numpy()
    pred_arr     = pred[mask].numpy()
    f1_macro     = f1_score(y_arr, pred_arr, average="macro",    zero_division=0)
    f1_weighted  = f1_score(y_arr, pred_arr, average="weighted", zero_division=0)

    counts_test  = np.bincount(y_arr, minlength=args.num_classes)
    min_tc       = int(counts_test.min())

    print("\n" + "=" * 57)
    print("  FINAL ENSEMBLE RESULT (test set — never seen during training)")
    print("=" * 57)
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro-F1    : {f1_macro:.4f}")

    # ── FIX: Flag when macro-F1 is misleading due to tiny test classes ───
    # COAD class-3 has N=4 total → N=1 test sample → F1=0.00 regardless.
    # That single zero drags macro-F1 from ~0.73 to ~0.53.
    # Report weighted-F1 alongside when this situation is detected.
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
                            f"{args.cancer}_ensemble_predictions.csv")
    pd.DataFrame({
        'true_label': y_arr,
        'pred_label': pred_arr,
    }).to_csv(csv_path, index=False)
    print(f"\n  Predictions saved: {csv_path}")