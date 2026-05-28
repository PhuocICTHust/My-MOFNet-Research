import argparse, os, pandas as pd, numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--project", required=True, choices=["TCGA-BRCA","TCGA-LGG","TCGA-STAD"])
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--train_ratio", type=float, default=0.6)
a = parser.parse_args()

tag = {"TCGA-BRCA":"BRCA","TCGA-LGG":"LGG","TCGA-STAD":"STAD"}[a.project]
os.makedirs("splits", exist_ok=True)

def load(name):
    p = f"data/raw/{a.project}_{name}.csv"
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}. Run the R script first.")
    df = pd.read_csv(p, index_col=0)
    df.columns = [c[:16] for c in df.columns]  # 统一条形码前16位
    return df

mrna, mirna, meth = load("mRNA_FPKM"), load("miRNA"), load("Meth450")
common = sorted(set(mrna.columns) & set(mirna.columns) & set(meth.columns))
rng = np.random.default_rng(a.seed); rng.shuffle(common)
n = int(len(common) * a.train_ratio)
train, test = common[:n], common[n:]

open(f"splits/{tag}_train.txt","w").write("\n".join(train))
open(f"splits/{tag}_test.txt","w").write("\n".join(test))
print(f"{tag}: train={len(train)}, test={len(test)} → splits/")
