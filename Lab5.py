"""

LAB 05 – TIỀN XỬ LÝ DỮ LIỆU + THỐNG KÊ MÔ TẢ                
Dataset : dulieuxettuyendaihoc.csv                                                                                              
FLOW:                                          
Load data → Missing → Heatmap → String→Số → Outlier (IQR)      
→ Mean/Median/Mode → Skewness/Kurtosis → Correlation            
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120

DATA_PATH = r"E:\MonDeepLearning\DataAnalystDeepLearning-main\DataAnalystDeepLearning-main\Data\dulieuxettuyendaihoc.csv"

# ══════════════════════════════════════════════════════════════════
# BƯỚC 1 – LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("=" * 65)
print("BƯỚC 1 – LOAD DATA")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
print(f"Kích thước: {df.shape[0]} dòng  x  {df.shape[1]} cột")
print("\n10 dòng đầu:")
print(df.head(10).to_string())
print("\nKiểu dữ liệu:")
print(df.dtypes)

# ══════════════════════════════════════════════════════════════════
# BƯỚC 2 – PHÂN LOẠI CỘT: ĐỊNH TÍNH vs ĐỊNH LƯỢNG
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 2 – PHÂN LOẠI DỮ LIỆU")
print("=" * 65)

# Cột định tính (String / categorical)
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
# Cột định lượng (số)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"Cột ĐỊNH TÍNH  (String): {cat_cols}")
print(f"Cột ĐỊNH LƯỢNG (Số)   : {num_cols}")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 3 – THỐNG KÊ & VISUALIZE DỮ LIỆU THIẾU (Heatmap)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 3 – THỐNG KÊ DỮ LIỆU THIẾU + HEATMAP")
print("=" * 65)

miss     = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
miss_df  = pd.DataFrame({'Thiếu': miss, 'Tỉ lệ (%)': miss_pct})
miss_df  = miss_df[miss_df['Thiếu'] > 0].sort_values('Tỉ lệ (%)', ascending=False)

if len(miss_df) > 0:
    print(miss_df)
else:
    print("Không có dữ liệu thiếu!")

# Heatmap TRƯỚC
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=ax)
ax.set_title("Heatmap – Dữ liệu thiếu (TRƯỚC xử lý)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("b3_heatmap_truoc.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════
# BƯỚC 4 – XỬ LÝ DỮ LIỆU THIẾU
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 4 – XỬ LÝ DỮ LIỆU THIẾU")
print("=" * 65)

# Cột định tính thiếu → điền mode
for col in cat_cols:
    n = df[col].isna().sum()
    if n > 0:
        m = df[col].mode()[0]
        df[col].fillna(m, inplace=True)
        print(f"  [{col}] String thiếu {n} → điền mode='{m}'")

# Cột định lượng thiếu → điền mean
for col in num_cols:
    n = df[col].isna().sum()
    if n > 0:
        m = df[col].mean()
        df[col].fillna(m, inplace=True)
        print(f"  [{col}] Số thiếu {n} → điền mean={m:.4f}")

total_missing = df.isnull().sum().sum()
print(f"\nTổng dữ liệu thiếu còn lại: {total_missing}")

# Heatmap SAU
fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=ax)
ax.set_title("Heatmap – Dữ liệu thiếu (SAU xử lý)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("b4_heatmap_sau.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════
# BƯỚC 5 – CHUYỂN STRING → SỐ (Encode Categorical)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 5 – CHUYỂN STRING → SỐ")
print("=" * 65)

df_encoded = df.copy()
encode_maps = {}

for col in cat_cols:
    unique_vals = df_encoded[col].unique()
    val_map = {v: i for i, v in enumerate(sorted([str(x) for x in unique_vals]))}
    df_encoded[col + '_num'] = df_encoded[col].astype(str).map(val_map)
    encode_maps[col] = val_map
    print(f"  [{col}] → [{col}_num]  mapping: {val_map}")

# Cập nhật danh sách cột số sau encode
num_cols_encoded = num_cols + [c + '_num' for c in cat_cols]
print(f"\nCột số sau encode: {num_cols_encoded}")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 6 – PHÁT HIỆN & XỬ LÝ OUTLIER (IQR)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 6 – OUTLIER (IQR Method)")
print("=" * 65)

# Minh hoạ với ví dụ thầy viết trên bảng
print("Ví dụ thầy: [5, 4, 7, 6, 3, 15, 40]")
demo = pd.Series([5, 4, 7, 6, 3, 15, 40])
Q1d  = demo.quantile(0.25)
Q3d  = demo.quantile(0.75)
IQRd = Q3d - Q1d
print(f"  Q1={Q1d}, Q3={Q3d}, IQR={IQRd}")
print(f"  Lower = Q1 - 1.5×IQR = {Q1d - 1.5*IQRd}")
print(f"  Upper = Q3 + 1.5×IQR = {Q3d + 1.5*IQRd}")
print(f"  Outlier: {demo[demo > Q3d + 1.5*IQRd].tolist()}\n")

# Áp dụng cho các cột điểm số
score_cols = [c for c in num_cols if c not in ['STT'] and df[c].dtype in [np.float64, np.int64]]

print(f"{'Cột':<8} {'Q1':>6} {'Q3':>6} {'IQR':>6} {'Lower':>7} {'Upper':>7} {'Outlier':>8}")
print("-" * 56)

bounds = {}
for col in score_cols:
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    bounds[col] = (lower, upper)
    print(f"{col:<8} {Q1:>6.2f} {Q3:>6.2f} {IQR:>6.2f} {lower:>7.2f} {upper:>7.2f} {n_out:>8}")

# Boxplot tất cả cột điểm
n_score = len(score_cols)
n_rows  = (n_score + 3) // 4
fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))
axes = axes.flatten()
for i, col in enumerate(score_cols):
    axes[i].boxplot(df[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    flierprops=dict(marker='o', color='red',
                                    markerfacecolor='red', markersize=4))
    axes[i].set_title(col, fontsize=10)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Boxplot – Phát hiện Outlier tất cả cột điểm",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("b6_boxplot_all.png", dpi=150)
plt.show()

# Capping outlier
df_clean = df.copy()
for col in score_cols:
    lo, hi = bounds[col]
    n = ((df_clean[col] < lo) | (df_clean[col] > hi)).sum()
    df_clean[col] = df_clean[col].clip(lower=lo, upper=hi)
    if n > 0:
        print(f"Capped '{col}': {n} outlier")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 7 – MEAN / MEDIAN / MODE + HÌNH DẠNG PHÂN PHỐI
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 7 – MEAN / MEDIAN / MODE")
print("=" * 65)

# Ví dụ cơ bản như trong notebook
speed = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
mode_r   = stats.mode(speed)
med_r    = np.median(speed)
mean_r   = np.mean(speed)
print(f"Ví dụ speed: {sorted(speed)}")
print(f"  Mode   = {mode_r.mode[0]}  (xuất hiện {mode_r.count[0]} lần)")
print(f"  Median = {med_r}")
print(f"  Mean   = {mean_r:.4f}")

# Phân tích cột T1 (hoặc cột số đầu tiên)
target_col = 'T1' if 'T1' in df_clean.columns else score_cols[0]
col_data   = df_clean[target_col].dropna()

t_mean   = col_data.mean()
t_median = col_data.median()
t_mode   = col_data.mode().values

print(f"\nPhân tích cột '{target_col}':")
print(f"  Mean   = {t_mean:.4f}")
print(f"  Median = {t_median:.4f}")
print(f"  Mode   = {t_mode}")

if abs(t_mean - t_median) < 0.5:
    conclusion = "Phân phối GẦN ĐỐI XỨNG (Normal)"
elif t_mean > t_median:
    conclusion = f"Mean({t_mean:.2f}) > Median({t_median:.2f}) → Lệch PHẢI (Right-Skewed)"
else:
    conclusion = f"Mean({t_mean:.2f}) < Median({t_median:.2f}) → Lệch TRÁI (Left-Skewed)"
print(f"\nNhận xét: {conclusion}")

# Histogram + Boxplot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Phân phối cột '{target_col}'", fontsize=14, fontweight='bold')
sns.histplot(col_data, kde=True, ax=axes[0], color='steelblue')
axes[0].axvline(t_mean,   color='red',    linestyle='--', label=f'Mean={t_mean:.2f}')
axes[0].axvline(t_median, color='green',  linestyle='--', label=f'Median={t_median:.2f}')
axes[0].axvline(t_mode[0],color='orange', linestyle='--', label=f'Mode={t_mode[0]:.2f}')
axes[0].legend()
axes[0].set_title("Histogram + KDE")
sns.boxplot(x=col_data, ax=axes[1], color='lightcoral')
axes[1].set_title("Boxplot")
plt.tight_layout()
plt.savefig(f"b7_dist_{target_col}.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════
# BƯỚC 8 – SKEWNESS & KURTOSIS
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 8 – SKEWNESS & KURTOSIS")
print("=" * 65)

skew_val = skew(col_data, bias=False)
kurt_val = kurtosis(col_data, bias=False)

print(f"Skewness = {skew_val:.6f}")
if   skew_val >  1: print("  → Lệch PHẢI đáng kể (> +1)")
elif skew_val < -1: print("  → Lệch TRÁI đáng kể (< -1)")
else:               print("  → Tương đối đối xứng (-1 ≤ skew ≤ 1)")

print(f"\nKurtosis = {kurt_val:.6f}")
if   kurt_val >  1: print("  → Đỉnh NHỌN (Leptokurtic > +1)")
elif kurt_val < -1: print("  → DẸET, đuôi mỏng (Platykurtic < -1)")
else:               print("  → Gần chuẩn (Mesokurtic)")

# Bảng tổng hợp tất cả cột điểm
print(f"\n{'Cột':<8} {'Mean':>8} {'Median':>8} {'Skewness':>10} {'Kurtosis':>10}")
print("-" * 50)
for col in score_cols:
    d = df_clean[col].dropna()
    print(f"{col:<8} {d.mean():>8.3f} {d.median():>8.3f}"
          f" {skew(d,bias=False):>10.4f} {kurtosis(d,bias=False):>10.4f}")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 9 – CORRELATION MATRIX (Heatmap)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 9 – CORRELATION & COVARIANCE MATRIX")
print("=" * 65)

corr = df_clean[score_cols].corr()
print("Correlation Matrix:")
print(corr.round(3))

sz = max(10, len(score_cols))
fig, ax = plt.subplots(figsize=(sz, sz-1))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5,
            square=True, ax=ax)
ax.set_title("Correlation Matrix – Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("b9_correlation_heatmap.png", dpi=150)
plt.show()

print("\nCovariance Matrix:")
print(df_clean[score_cols].cov().round(3))

# ══════════════════════════════════════════════════════════════════
# BƯỚC 10 – THỐNG KÊ MÔ TẢ TỔNG HỢP + HISTOGRAM TẤT CẢ CỘT
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 10 – THỐNG KÊ MÔ TẢ TỔNG HỢP")
print("=" * 65)
print(df_clean[score_cols].describe().round(3).to_string())

n = len(score_cols)
n_rows = (n + 3) // 4
fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4*n_rows))
axes = axes.flatten()
for i, col in enumerate(score_cols):
    d = df_clean[col].dropna()
    sns.histplot(d, kde=True, ax=axes[i], color='steelblue')
    axes[i].axvline(d.mean(),   color='red',   linestyle='--', linewidth=1.2, label='Mean')
    axes[i].axvline(d.median(), color='green', linestyle='--', linewidth=1.2, label='Median')
    axes[i].set_title(col, fontsize=10)
    axes[i].legend(fontsize=7)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Histogram + KDE – Tất cả cột điểm",
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("b10_all_histograms.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ LAB 05 HOÀN THÀNH!")
print(f"Dataset sạch: {df_clean.shape[0]} dòng  x  {df_clean.shape[1]} cột")