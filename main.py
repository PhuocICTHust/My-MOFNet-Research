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
parser.add_argument("--cancer",         required=True)
parser.add_argument("--data_path",      required=True)
parser.add_argument("--epochs",         type=int,   default=150)
parser.add_argument("--nhid",           type=int,   default=64)
parser.add_argument("--num_classes",    type=int,   default=5)
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

    # FIX Bug 2: explicit key access — robust against dict insertion order changes
    d1 = data['mRNA']
    d2 = data['miRNA']
    d3 = data['Methy']
    d4 = data['CNV']

    optimizer.zero_grad()

    # FIX Bug 1: model returns (logits, entropy) — unpack correctly
    logits, entropy = model(d1, d2, d3, d4)

    mask = mask.to(logits.device)
    y    = d1.y.to(logits.device)

    focal_loss = criterion(logits[mask], y[mask])

    # FIX Bug 1 (entropy sign): SUBTRACT entropy to MAXIMIZE it.
    # Goal: encourage uniform attention across 4 omics → prevent modality collapse.
    loss = focal_loss - args.entropy_weight * entropy

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    optimizer.step()

    return loss.item(), focal_loss.item()


# ─────────────────────────────
# TEST
# ─────────────────────────────
def test(model, data, mask):
    model.eval()

    # FIX Bug 2: explicit key access
    d1 = data['mRNA']
    d2 = data['miRNA']
    d3 = data['Methy']
    d4 = data['CNV']

    with torch.no_grad():
        # FIX Bug 1: unpack tuple correctly
        logits, _ = model(d1, d2, d3, d4)
        pred      = logits.argmax(dim=1)

    # Force CPU before numpy/sklearn
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
    print(f"\n{'='*45}")
    print(f"  Seed {seed}")
    print(f"{'='*45}")
    set_seed(seed)

    data   = load_graph()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    for k in data:
        data[k] = data[k].to(device)

    model = PanCancerModel(args, data).to(device)

    # Class weights (balanced) — clamp to avoid extreme gradients with minority class
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

    train_mask = data["mRNA"].train_mask.to(device)
    test_mask  = data["mRNA"].test_mask.to(device)

    best_f1    = 0.0
    last_k     = []   # last-10 logits for ensemble
    f1_history = []   # ← F1 theo từng epoch để plot

    # FIX Bug 3: create checkpoint directory and save best model per seed
    ckpt_dir  = os.path.join(args.data_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.cancer}_seed{seed}_best.pth")

    for epoch in range(args.epochs):
        loss, focal = train(model, data, optimizer, criterion, train_mask)
        acc, f1, logits_cpu, attn = test(model, data, test_mask)

        # ← Ghi lại F1 sau mỗi epoch
        f1_history.append(f1)

        # FIX Bug 3: save best checkpoint when F1 improves
        if f1 > best_f1:
            best_f1 = f1
            torch.save({
                'epoch':             epoch,
                'model_state_dict':  model.state_dict(),
                'f1':                f1,
                'acc':               acc,
                'args':              vars(args),
                'attention_weights': attn,
                'y':                 data['mRNA'].y.cpu(),
                'omics_names':       ['mRNA', 'miRNA', 'Methy', 'CNV'],
            }, ckpt_path)

        # Collect last 10 epochs for ensemble
        if epoch >= args.epochs - 10:
            last_k.append(logits_cpu)

        if epoch % 10 == 0:
            print(f"  Ep {epoch:3d} | loss={loss:.4f} focal={focal:.4f} | "
                  f"acc={acc:.4f} macro-F1={f1:.4f}")

    print(f"  → Best macro-F1 (seed {seed}): {best_f1:.4f}")
    print(f"  → Checkpoint saved: {ckpt_path}")

    # ── Plot F1 curve cho seed này ──────────────────────────────────────
    plt.figure(figsize=(8, 4))
    plt.plot(f1_history, linewidth=1.5, color='steelblue')

    # Đánh dấu epoch tốt nhất
    best_epoch = int(np.argmax(f1_history))
    plt.axvline(x=best_epoch, color='tomato', linestyle='--', linewidth=1,
                label=f'Best epoch {best_epoch} (F1={best_f1:.4f})')

    plt.xlabel('Epoch')
    plt.ylabel('Test Macro-F1')
    plt.title(f'{args.cancer} — Seed {seed}')
    plt.legend(fontsize=9)
    plt.tight_layout()

    plot_path = os.path.join(args.data_path, f"f1_curve_{args.cancer}_seed{seed}.png")
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  → F1 curve saved: {plot_path}")
    # ────────────────────────────────────────────────────────────────────

    ensemble_logits = torch.stack(last_k).mean(dim=0)  # [N, C]
    return ensemble_logits, data, f1_history


# ─────────────────────────────
# MAIN
# ─────────────────────────────
if __name__ == "__main__":

    seeds         = [777, 42, 1234, 2024, 999]
    all_logits    = []
    all_f1_curves = []   # ← tất cả F1 curves để plot tổng
    final_data    = None

    for s in seeds:
        logits, data, f1_hist = run_one_seed(s)
        all_logits.append(logits)
        all_f1_curves.append(f1_hist)
        final_data = data

    # ── Plot tổng: tất cả 5 seeds trên cùng 1 figure ───────────────────
    plt.figure(figsize=(10, 5))
    colors = ['steelblue', 'darkorange', 'seagreen', 'mediumpurple', 'crimson']
    for idx, (hist, seed) in enumerate(zip(all_f1_curves, seeds)):
        plt.plot(hist, linewidth=1.2, alpha=0.8,
                 color=colors[idx], label=f'Seed {seed}')

    # Mean curve
    mean_curve = np.mean(all_f1_curves, axis=0)
    plt.plot(mean_curve, linewidth=2.5, color='black',
             linestyle='--', label='Mean')

    plt.xlabel('Epoch')
    plt.ylabel('Test Macro-F1')
    plt.title(f'{args.cancer} — All seeds F1 curves')
    plt.legend(fontsize=9)
    plt.tight_layout()

    summary_plot = os.path.join(args.data_path,
                                f"f1_curve_{args.cancer}_all_seeds.png")
    plt.savefig(summary_plot, dpi=120)
    plt.close()
    print(f"\n  → Summary F1 curve saved: {summary_plot}")
    # ────────────────────────────────────────────────────────────────────

    # ── ENSEMBLE ────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("  FINAL ENSEMBLE RESULT")
    print("="*50)

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
