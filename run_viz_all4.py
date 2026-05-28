"""
Vẽ omics attention của cả 4 cancer types trong 1 figure — dùng cho paper.
Chạy: python run_viz_all4.py
Output: E:\Cancer-classification-dataset\omics_attention_4cancers.png
"""
import os, torch, numpy as np, matplotlib.pyplot as plt

CKPT_DIR = r'E:\Cancer-classification-dataset\checkpoints'
SEEDS    = [777, 42, 1234, 2024, 999]
CANCERS  = ['BRCA', 'COAD', 'GBM', 'OV']
OMICS    = ['mRNA', 'miRNA', 'Methy', 'CNV']
OUT_DIR  = r'E:\Cancer-classification-dataset'
COLORS   = ['#E07B6A', '#6BBFCF', '#5DBFA8', '#8B9DC3']


def load_attn(cancer):
    all_attn, loaded = [], []
    for s in SEEDS:
        path = os.path.join(CKPT_DIR, f'{cancer}_seed{s}_best.pth')
        if not os.path.exists(path):
            print(f'  [{cancer}] SKIP seed {s} — file not found')
            continue
        ck = torch.load(path, weights_only=False)
        if 'attention_weights' not in ck:
            print(f'  [{cancer}] SKIP seed {s} — no attention_weights key')
            continue
        attn = ck['attention_weights']
        if isinstance(attn, torch.Tensor):
            attn = attn.cpu().numpy()
        all_attn.append(attn.mean(axis=0))
        loaded.append(s)
    return np.array(all_attn), loaded


results = {}
for c in CANCERS:
    print(f'\n=== {c} ===')
    mat, seeds = load_attn(c)
    if len(mat) == 0:
        print(f'  ERROR: no data for {c}, skipping')
        continue
    results[c] = {
        'means':  mat.mean(axis=0),
        'stds':   mat.std(axis=0),
        'n_seeds': len(seeds),
        'seeds':   seeds,
    }
    for o, m, s in zip(OMICS, results[c]['means'], results[c]['stds']):
        print(f'  {o:8s}: {m:.4f} ± {s:.4f}')

if not results:
    print('ERROR: no data loaded at all.')
    raise SystemExit(1)

# ── PLOT: 1 row × N cols ──────────────────────────────────────────
n = len(results)
fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8), sharey=False)
if n == 1:
    axes = [axes]

for ax, cancer in zip(axes, results):
    r     = results[cancer]
    means = r['means']
    stds  = r['stds']
    bars  = ax.bar(OMICS, means, yerr=stds, capsize=6,
                   color=COLORS, alpha=0.85, width=0.55,
                   error_kw=dict(elinewidth=1.2, capthick=1.2, ecolor='#333'))
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.006,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9.5)
    ax.set_title(f'{cancer}\n({r["n_seeds"]} seeds)', fontsize=11, fontweight='normal')
    ax.set_ylabel('Mean Attention Weight' if ax == axes[0] else '', fontsize=10)
    ax.set_ylim(0, max(means + stds) * 1.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=9)

fig.suptitle('Omics Attention Weights — LightGATEncoder', fontsize=13, y=1.02)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, 'omics_attention_4cancers.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved: {out_path}')
