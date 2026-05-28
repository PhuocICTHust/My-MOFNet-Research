"""
Visualize omics attention weights — GBM
Chạy: python run_viz_GBM.py
Output: GBM_omics_attention.png
"""
import os, torch, numpy as np, matplotlib.pyplot as plt

CKPT_DIR   = r'E:\Cancer-classification-dataset\checkpoints'
SEEDS      = [777, 42, 1234, 2024, 999]
CANCER     = 'GBM'
OMICS      = ['mRNA', 'miRNA', 'Methy', 'CNV']
OUT_DIR    = r'E:\Cancer-classification-dataset'

def load_attn(ckpt_dir, cancer, seeds):
    all_attn = []
    loaded   = []
    for s in seeds:
        path = os.path.join(ckpt_dir, f'{cancer}_seed{s}_best.pth')
        if not os.path.exists(path):
            print(f'  [SKIP] not found: {path}')
            continue
        ck = torch.load(path, weights_only=False)
        if 'attention_weights' not in ck:
            print(f'  [SKIP] no attention_weights key in seed {s}')
            continue
        attn = ck['attention_weights']
        if isinstance(attn, torch.Tensor):
            attn = attn.cpu().numpy()
        mean_per_sample = attn.mean(axis=0)
        all_attn.append(mean_per_sample)
        loaded.append(s)
        print(f'  seed {s}: {np.round(mean_per_sample, 4)}')
    return np.array(all_attn), loaded

print(f'\n=== {CANCER} omics attention ===')
attn_matrix, loaded_seeds = load_attn(CKPT_DIR, CANCER, SEEDS)

if len(attn_matrix) == 0:
    print('ERROR: no checkpoints loaded.')
    raise SystemExit(1)

means = attn_matrix.mean(axis=0)
stds  = attn_matrix.std(axis=0)

print(f'\nLoaded {len(loaded_seeds)} seeds: {loaded_seeds}')
for o, m, s in zip(OMICS, means, stds):
    print(f'  {o:8s}: {m:.4f} ± {s:.4f}')

colors = ['#E07B6A', '#6BBFCF', '#5DBFA8', '#8B9DC3']
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(OMICS, means, yerr=stds, capsize=6,
              color=colors, alpha=0.85, width=0.55,
              error_kw=dict(elinewidth=1.2, capthick=1.2, ecolor='#333'))

for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

ax.set_title(f'{CANCER} — Omics Attention (LightGATEncoder)\n'
             f'({len(loaded_seeds)} seeds)', fontsize=12)
ax.set_ylabel('Mean Attention Weight', fontsize=11)
ax.set_ylim(0, max(means + stds) * 1.25)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, f'{CANCER}_omics_attention.png')
plt.savefig(out_path, dpi=150)
plt.close()
print(f'\nSaved: {out_path}')
