This folder contains sample barcode lists (train/test) for reproducibility.
Run:
  source("scripts/get_tcga_data.R"); get_all("TCGA-BRCA")
  python scripts/preprocess_and_split.py --project TCGA-BRCA
