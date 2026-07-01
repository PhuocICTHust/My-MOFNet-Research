import os
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA


def run_ml_baseline(cancer_type, split_seed, data_path):
    # 1. Load trực tiếp file .pt đã được prepare_graph.py tạo ra
    # Lưu ý: Tên file có thể có hoặc không có _s{seed} tùy bạn đang dùng phiên bản nào
    pt_file = os.path.join(data_path, f"{cancer_type}_graph.pt")
    if not os.path.exists(pt_file):
        pt_file = os.path.join(data_path, f"{cancer_type}_graph_s{split_seed}.pt")

    print(f"\n[INFO] Loading {pt_file}...")
    dataset = torch.load(pt_file, weights_only=False)

    # 2. Lấy features của 4 omics và nối ngang lại (Early Integration)
    X_mrna = dataset['mRNA'].x.numpy()
    X_mirna = dataset['miRNA'].x.numpy()
    X_methy = dataset['Methy'].x.numpy()
    X_cnv = dataset['CNV'].x.numpy()

    # Nối (Concatenate) 4 omics thành 1 vector dài cho mỗi bệnh nhân
    X_concat = np.hstack([X_mrna, X_mirna, X_methy, X_cnv])
    y = dataset['mRNA'].y.numpy()

    # 3. Lấy Masks và GỘP TRAIN + VAL
    train_mask = dataset['mRNA'].train_mask.numpy()
    val_mask = dataset['mRNA'].val_mask.numpy()
    test_mask = dataset['mRNA'].test_mask.numpy()

    # Kỹ thuật gộp: Dùng phép toán OR (|) để lấy cả Train và Val
    train_val_mask = train_mask | val_mask

    X_train = X_concat[train_val_mask]
    y_train = y[train_val_mask]

    X_test = X_concat[test_mask]
    y_test = y[test_mask]

    print(f"-> Shape ban đầu: Train+Val={X_train.shape}, Test={X_test.shape}")

    # 4. Giảm chiều bằng PCA (BẮT BUỘC cho ML để tránh lời nguyền số chiều)
    pca = PCA(n_components=64, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"-> Sau PCA: Train+Val={X_train_pca.shape}, Test={X_test_pca.shape}")

    # 5. Khởi tạo và Huấn luyện các mô hình Baseline
    models = {
        "SVM (RBF)": SVC(kernel='rbf', class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }

    results = {}
    for name, model in models.items():
        # Train
        model.fit(X_train_pca, y_train)
        # Predict trên tập Test hoàn toàn mù
        y_pred = model.predict(X_test_pca)

        # Đánh giá
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results[name] = {"Acc": acc, "Macro-F1": macro_f1, "Weighted-F1": weighted_f1}
        print(f"   [{name}] Acc: {acc:.4f} | Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f}")

    return results


if __name__ == "__main__":
    # Cấu hình đường dẫn của bạn
    DATA_PATH = r"E:\Cancer-classification-dataset"
    CANCER = "OV"

    # Chạy vòng lặp qua 5 seed: 0, 1, 2, 3, 4
    for seed in range(5):
        print(f"\n{'=' * 40}\n Chạy Baseline ML cho Seed: {seed}\n{'=' * 40}")
        run_ml_baseline(CANCER, split_seed=seed, data_path=DATA_PATH)