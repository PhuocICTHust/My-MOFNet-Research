import argparse
import os
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from models import PanCancerModel, FocalLoss


# ─────────────────────────────
# ARGS
# ─────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--cancer",         required=True,
                    choices=["BRCA", "COAD", "GBM", "LGG", "OV"],
                    help="Cancer type — must match the *_graph.pt file produced by prepare_graph.py.")
parser.add_argument("--data_path",      required=True)
parser.add_argument("--epochs",         type=int,   default=150)
parser.add_argument("--nhid",           type=int,   default=64)
parser.add_argument("--num_classes",    type=int,   default=0,
                    help="Number of output classes. "
                         "0 = auto-detect from data (strongly recommended). "
                         "BRCA/GBM=5, COAD/OV=4, LGG=3.")
parser.add_argument("--dropout_ratio",  type=float, default=0.3)
parser.add_argument("--lr",             type=float, default=5e-4)
parser.add_argument("--weight_decay",   type=float, default=5e-4)
parser.add_argument("--focal_gamma",    type=float, default=2.0)
parser.add_argument("--entropy_weight", type=float, default=0.01)
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
# TRAIN
# ─────────────────────────────
def train(model, data, optimizer, criterion, mask):
    model.train()

    d1, d2, d3, d4 = data['mRNA'], data['miRNA'], data['Methy'], data['CNV']

    optimizer.zero_grad()
    logits, entropy = model(d1, d2, d3, d4)

    mask = mask.to(logits.device)
    y    = d1.y.to(logits.device)

    focal_loss = criterion(logits[mask], y[mask])

    # Maximize entropy → subtract from loss.
    # Goal: prevent attention from collapsing onto a single omic modality.
    loss = focal_loss - args.entropy_weight * entropy

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()

    return loss.item(), focal_loss.item()


# ─────────────────────────────
# EVALUATE (works for any mask: val or test)
# ─────────────────────────────
def evaluate(model, data, mask):
    """
    Renamed from test() → evaluate() to be explicit about what mask is used.
    Call with val_mask during training (checkpoint selection).
    Call with test_mask for final reporting only.
    """
    model.eval()

    d1, d2, d3, d4 = data['mRNA'], data['miRNA'], data['Methy'], data['CNV']

    with torch.no_grad():
        logits, _ = model(d1, d2, d3, d4)
        pred      = logits.argmax(dim=1)

    y    = d1.y.cpu()
    pred = pred.cpu()
    mask = mask.cpu()

    acc = accuracy_score(y[mask], pred[mask])
    f1  = f1_score(y[mask], pred[mask], average="macro", zero_division=0)

    attn = model.get_attention_weights().cpu()  # [N, 4]

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

    # Class weights (balanced, clamped to avoid extreme gradients)
    y_np    = data["mRNA"].y.cpu().numpy()
    classes = np.unique(y_np)
    w       = compute_class_weight("balanced", classes=classes, y=y_np)
    w       = np.clip(w, 0.5, 5.0)
    w_t     = torch.tensor(w, dtype=torch.float32).to(device)
    print(f"  Class weights (clamped): {np.round(w, 4)}")

    criterion = FocalLoss(gamma=args.focal_gamma, alpha=w_t)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # FIX #1: CosineAnnealingLR scheduler — zero-risk, high impact.
    # Decays LR smoothly from args.lr → eta_min over T_max epochs.
    # Avoids plateau in the flat region of constant LR.
    # Typically adds +0.01–0.03 macro-F1 with no other changes.
    # T_max = args.epochs: one full cosine cycle (no restarts).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    train_mask = data["mRNA"].train_mask.to(device)
    val_mask   = data["mRNA"].val_mask.to(device)     # FIX #2: val_mask from 70/10/20 split
    test_mask  = data["mRNA"].test_mask.to(device)

    best_val_f1       = 0.0
    last_k_logits     = []        # last-10 epochs for ensemble (on full graph)
    val_f1_history    = []        # per-epoch val F1 for plot

    ckpt_dir  = os.path.join(args.data_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.cancer}_seed{seed}_best.pth")

    for epoch in range(args.epochs):
        loss, focal = train(model, data, optimizer, criterion, train_mask)
        scheduler.step()    # advance LR schedule after each epoch

        # FIX #2: Evaluate on VAL for checkpoint selection (no test leakage).
        # Also monitor test for logging — but checkpoint is driven by val F1.
        val_acc,  val_f1,  _,          _    = evaluate(model, data, val_mask)
        test_acc, test_f1, logits_cpu,  attn = evaluate(model, data, test_mask)

        val_f1_history.append(val_f1)

        # Save checkpoint when VAL F1 improves (not test F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch':             epoch,
                'model_state_dict':  model.state_dict(),
                'val_f1':            val_f1,
                'test_f1':           test_f1,    # informational only, not used for selection
                'args':              vars(args),
                'attention_weights': attn,
                'y':                 data['mRNA'].y.cpu(),
                'omics_names':       ['mRNA', 'miRNA', 'Methy', 'CNV'],
            }, ckpt_path)

        # Collect last 10 epochs for cross-seed ensemble
        if epoch >= args.epochs - 10:
            last_k_logits.append(logits_cpu)

        if epoch % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Ep {epoch:3d} | loss={loss:.4f} focal={focal:.4f} | "
                  f"val_acc={val_acc:.4f} val_F1={val_f1:.4f} | "
                  f"test_acc={test_acc:.4f} test_F1={test_f1:.4f} | "
                  f"lr={lr_now:.2e}")

    print(f"  → Best val macro-F1 (seed {seed}): {best_val_f1:.4f}")
    print(f"  → Checkpoint saved: {ckpt_path}")

    # ── Val F1 curve for this seed ───────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(val_f1_history, linewidth=1.5, color='steelblue', label='Val Macro-F1')

    best_epoch = int(np.argmax(val_f1_history))
    plt.axvline(x=best_epoch, color='tomato', linestyle='--', linewidth=1,
                label=f'Best epoch {best_epoch} (val F1={best_val_f1:.4f})')

    plt.xlabel('Epoch')
    plt.ylabel('Val Macro-F1')
    plt.title(f'{args.cancer} — Seed {seed} (val)')
    plt.legend(fontsize=9)
    plt.tight_layout()

    plot_path = os.path.join(args.data_path, f"f1_curve_{args.cancer}_seed{seed}.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  → Val F1 curve saved: {plot_path}")

    ensemble_logits = torch.stack(last_k_logits).mean(dim=0)   # [N, C]
    return ensemble_logits, data, val_f1_history


# ─────────────────────────────
# MAIN
# ─────────────────────────────
if __name__ == "__main__":

    # Auto-detect num_classes from graph file.
    # Critical: avoids RuntimeError in FocalLoss for COAD/OV (4 classes) and LGG (3 classes)
    # when a hard-coded num_classes=5 was used previously.
    if args.num_classes == 0:
        _tmp = load_graph()
        args.num_classes = int(_tmp["mRNA"].y.max().item()) + 1
        del _tmp

    print(f"\n  Cancer      : {args.cancer}")
    print(f"  num_classes : {args.num_classes}  (BRCA/GBM=5, COAD/OV=4, LGG=3)")

    seeds         = [777, 42, 1234, 2024, 999]
    all_logits    = []
    all_f1_curves = []
    final_data    = None

    for s in seeds:
        logits, data, f1_hist = run_one_seed(s)
        all_logits.append(logits)
        all_f1_curves.append(f1_hist)
        final_data = data

    # ── Summary plot: all 5 seeds val F1 curves ─────────────────────────
    plt.figure(figsize=(10, 5))
    colors = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple', 'crimson']
    for idx, (hist, seed) in enumerate(zip(all_f1_curves, seeds)):
        plt.plot(hist, linewidth=1.2, alpha=0.8,
                 color=colors[idx], label=f'Seed {seed}')

    mean_curve = np.mean(all_f1_curves, axis=0)
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
    print(f"\n  → Summary val F1 curve saved: {summary_plot}")

    # ── ENSEMBLE ─────────────────────────────────────────────────────────
    # Cross-seed ensemble: average logits from last 10 epochs of each seed.
    # Evaluated ONLY on test_mask (not used during training at all).
    print("\n" + "=" * 55)
    print("  FINAL ENSEMBLE RESULT (test set — never seen during training)")
    print("=" * 55)

    ensemble = torch.stack(all_logits).mean(dim=0)  # [N, C]
    pred     = ensemble.argmax(dim=1)

    y    = final_data["mRNA"].y.cpu()
    mask = final_data["mRNA"].test_mask.cpu()

    acc = accuracy_score(y[mask], pred[mask])
    f1  = f1_score(y[mask], pred[mask], average="macro", zero_division=0)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Macro-F1 : {f1:.4f}")

    y_all     = y.numpy()
    counts    = np.bincount(y_all, minlength=args.num_classes)
    cls_names = [f"Class{i}({counts[i]})" for i in range(args.num_classes)]

    print("  Per-class report:")
    print(classification_report(
        y[mask].numpy(), pred[mask].numpy(),
        target_names=cls_names,
        zero_division=0
    ))

    # Save ensemble predictions to CSV
    csv_path = os.path.join(args.data_path, f"{args.cancer}_ensemble_predictions.csv")
    pd.DataFrame({
        'true_label': y[mask].numpy(),
        'pred_label': pred[mask].numpy(),
    }).to_csv(csv_path, index=False)
    print(f"\n  Predictions saved: {csv_path}")
