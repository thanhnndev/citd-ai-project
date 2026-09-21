"""
Script tự động sinh trọn bộ biểu đồ trực quan (EDA, Feature Importance, Confusion Matrix, ROC Curves)
để chèn vào Báo cáo Word, Slide thuyết trình và Notebook.
"""
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier

# Cấu hình đường dẫn
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed"
OUTPUT_DIR = ROOT / "outputs"
STAGE3_DIR = OUTPUT_DIR / "holdout/stage3"
STAGE4_DIR = OUTPUT_DIR / "holdout/stage4"
OOF_DIR = OUTPUT_DIR / "step4_thread1/catboost_training"
MODEL_PATH = OUTPUT_DIR / "holdout/stage2/catboost_final_holdout_run1.cbm"
FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Cài đặt style đẹp mắt cho matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.titlesize'] = 14

FEATURES = [
    'breakeven_R', 'atr14_pct', 'atr_ratio_14_90', 'vol20', 'vol200', 'vol_ratio_20_200',
    'range_pct', 'dist_ema20_atr', 'dist_ema50_atr', 'dist_ema200_atr', 'ret20_atr', 'ret50_atr',
    'pos_in_range50', 'dist_hh20_atr', 'mom14', 'mom_diff', 'vwap_dist_atr', 'vwap_slope_atr',
    'rsi14', 'vol_ratio_volume', 'gap_open_atr', 'hour', 'dow'
]

def generate_figure1_class_distribution():
    """Hình 1: Phân bố nhãn trên tập Train/Dev và Holdout"""
    print("[1/5] Đang vẽ 01_class_distribution.png...")
    df_train = pd.read_csv(DATA_DIR / "dataset_catboost.csv")
    df_holdout = pd.read_csv(STAGE3_DIR / "holdout_scored.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    
    # Train/Dev set
    train_counts = df_train['label'].value_counts().sort_index()
    train_pcts = train_counts / len(df_train) * 100
    colors = ['#4A90E2', '#50E3C2']
    
    bars1 = axes[0].bar(['Thua / Hết giờ (0)', 'Thắng (1)'], train_counts, color=['#E74C3C', '#2ECC71'], width=0.55, edgecolor='black', linewidth=1)
    axes[0].set_title(f"Tập Train & Dev (N = {len(df_train):,} mẫu)", fontweight='bold', pad=15)
    axes[0].set_ylabel("Số lượng mẫu giao dịch")
    axes[0].set_ylim(0, max(train_counts) * 1.18)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    for bar, count, pct in zip(bars1, train_counts, train_pcts):
        y = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, y + len(df_train)*0.02, f"{count:,}\n({pct:.1f}%)", ha='center', va='bottom', fontweight='bold')

    # Holdout set
    holdout_counts = df_holdout['label'].value_counts().sort_index()
    holdout_pcts = holdout_counts / len(df_holdout) * 100
    
    bars2 = axes[1].bar(['Thua / Hết giờ (0)', 'Thắng (1)'], holdout_counts, color=['#E74C3C', '#2ECC71'], width=0.55, edgecolor='black', linewidth=1)
    axes[1].set_title(f"Tập Sealed Holdout (N = {len(df_holdout):,} mẫu)", fontweight='bold', pad=15)
    axes[1].set_ylabel("Số lượng mẫu giao dịch")
    axes[1].set_ylim(0, max(holdout_counts) * 1.18)
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    for bar, count, pct in zip(bars2, holdout_counts, holdout_pcts):
        y = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, y + len(df_holdout)*0.02, f"{count:,}\n({pct:.1f}%)", ha='center', va='bottom', fontweight='bold')

    plt.suptitle("PHÂN BỐ TỶ LỆ NHÃN TRIPLE-BARRIER TRÊN TẬP HUẤN LUYỆN VÀ TẬP KIỂM ĐỊNH", fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_class_distribution.png", bbox_inches='tight')
    plt.close()

def generate_figure2_correlation_heatmap():
    """Hình 2: Ma trận tương quan giữa 23 đặc trưng kỹ thuật"""
    print("[2/5] Đang vẽ 02_feature_correlation_heatmap.png...")
    df = pd.read_csv(DATA_DIR / "dataset_catboost.csv", usecols=FEATURES)
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(14, 11), dpi=150)
    cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04, label="Hệ số tương quan Pearson")

    ticks = range(len(FEATURES))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(FEATURES, rotation=90, ha='left', fontsize=9)
    ax.set_yticklabels(FEATURES, fontsize=9)
    ax.set_title("MA TRẬN TƯƠNG QUAN GIỮA 23 ĐẶC TRƯNG ĐẦU VÀO (FEATURES HEATMAP)", fontweight='bold', pad=30)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_feature_correlation_heatmap.png", bbox_inches='tight')
    plt.close()

def generate_figure3_feature_importance():
    """Hình 3: Xếp hạng tầm quan trọng của các đặc trưng trong mô hình CatBoost"""
    print("[3/5] Đang vẽ 03_catboost_feature_importance.png...")
    if not MODEL_PATH.exists():
        print("Không tìm thấy model, bỏ qua hình 3")
        return
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    importance = model.get_feature_importance()
    feat_df = pd.DataFrame({'feature': FEATURES, 'importance': importance}).sort_values('importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    bars = ax.barh(feat_df['feature'], feat_df['importance'], color='#3498DB', edgecolor='black', height=0.65)
    ax.set_xlabel("Mức độ quan trọng (%)")
    ax.set_title("XẾP HẠNG TẦM QUAN TRỌNG CỦA 23 ĐẶC TRƯNG (CATBOOST FEATURE IMPORTANCE)", fontweight='bold', pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.15, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va='center', fontsize=9)

    ax.set_xlim(0, max(feat_df['importance']) * 1.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_catboost_feature_importance.png", bbox_inches='tight')
    plt.close()

def generate_figure4_confusion_matrix():
    """Hình 4: Ma trận nhầm lẫn của CatBoost trên tập Holdout"""
    print("[4/5] Đang vẽ 04_confusion_matrix_holdout.png...")
    df_holdout = pd.read_csv(STAGE3_DIR / "holdout_scored.csv")
    y_true = df_holdout['label'].to_numpy()
    # Phân loại theo ngưỡng cut-off 0.5
    y_pred = (df_holdout['probability'].to_numpy() >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    total = len(y_true)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(cax, fraction=0.046, pad=0.04)

    classes = ['Thua / Hết giờ (0)', 'Thắng (1)']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)

    # Điền giá trị vào từng ô
    thresh = cm.max() / 2.
    labels_desc = [
        ["TN (Đoán đúng lệnh Thua)", "FP (Báo nhầm - Mất tiền)"],
        ["FN (Bỏ sót - Không mất tiền)", "TP (Đoán đúng lệnh Thắng)"]
    ]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = val / total * 100
            desc = labels_desc[i][j]
            color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:,}\n({pct:.1f}%)\n\n[{desc}]",
                    ha="center", va="center", color=color, fontsize=10, fontweight='bold')

    ax.set_ylabel('Nhãn thực tế (True Label)', fontweight='bold')
    ax.set_xlabel('Nhãn dự đoán (Predicted Label @ 0.5)', fontweight='bold')
    ax.set_title(f"MA TRẬN NHẦM LẪN TRÊN TẬP HOLDOUT (N = {total:,} MẪU)", fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_confusion_matrix_holdout.png", bbox_inches='tight')
    plt.close()

def generate_figure5_roc_curves():
    """Hình 5: So sánh đường cong ROC của 4 cách chia dữ liệu và tập Holdout"""
    print("[5/5] Đang vẽ 05_roc_curves_comparison.png...")
    methods = [
        ("oof_random_kfold.csv", "Cách 1 — Random K-Fold", "#E74C3C", 2),
        ("oof_grouped_kfold.csv", "Cách 1b — Grouped K-Fold", "#E67E22", 2),
        ("oof_walk_forward.csv", "Cách 2 — Walk-Forward", "#F1C40F", 2),
        ("oof_purged_walk_forward.csv", "Cách 3 — WF + Purge/Embargo", "#2980B9", 2.5),
    ]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    # Vẽ 4 cách chia
    for filename, label, color, lw in methods:
        file_path = OOF_DIR / filename
        if file_path.exists():
            df = pd.read_csv(file_path)
            # Chỉ lấy khúc 2-5
            if 'fold' in df.columns:
                df = df[df['fold'] >= 1]
            fpr, tpr, _ = roc_curve(df['label'], df['probability'])
            auc = roc_auc_score(df['label'], df['probability'])
            ax.plot(fpr, tpr, color=color, lw=lw, label=f"{label} (AUC = {auc:.4f})")

    # Vẽ Holdout
    holdout_file = STAGE3_DIR / "holdout_scored.csv"
    if holdout_file.exists():
        df_h = pd.read_csv(holdout_file)
        fpr, tpr, _ = roc_curve(df_h['label'], df_h['probability'])
        auc_h = roc_auc_score(df_h['label'], df_h['probability'])
        ax.plot(fpr, tpr, color="#2ECC71", lw=3, linestyle="--", label=f"Holdout Niêm Phong (AUC = {auc_h:.4f})")

    # Đường chéo ngẫu nhiên 0.5
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle=':', label='Đoán ngẫu nhiên (AUC = 0.5000)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (Tỷ lệ báo nhầm)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Tỷ lệ bắt đúng)', fontweight='bold')
    ax.set_title('SO SÁNH ĐƯỜNG CONG ROC QUA 4 CÁCH CHIA VÀ TẬP HOLDOUT', fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_roc_curves_comparison.png", bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    generate_figure1_class_distribution()
    generate_figure2_correlation_heatmap()
    generate_figure3_feature_importance()
    generate_figure4_confusion_matrix()
    generate_figure5_roc_curves()
    print("HOÀN THÀNH: Tất cả 5 hình ảnh đã được lưu vào:", FIGURES_DIR)
