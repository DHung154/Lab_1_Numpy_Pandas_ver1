"""
                                                         
FLOW đầy đủ (theo bảng thầy):                                   
Load → Phân loại → Missing + Heatmap → String→Số               
 → Outlier (IQR) → TBM / XL / KQXT → MinMax Scale               
 → Input/Target → Lưu file                                       

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 110

DATA_PATH = r"E:\MonDeepLearning\DataAnalystDeepLearning-main\DataAnalystDeepLearning-main\Data\dulieuxettuyendaihoc.csv"
OUT_PATH  = r"E:\MonDeepLearning\processed_dulieuxettuyendaihoc.csv"

# ══════════════════════════════════════════════════════════════════
# BƯỚC 1 – LOAD DATA & XEM TỔNG QUAN
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("BƯỚC 1 – LOAD DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"Kích thước: {df.shape[0]} dòng  x  {df.shape[1]} cột")
print("\n--- 10 DÒNG ĐẦU ---")
print(df.head(10).to_string())
print("\n--- 10 DÒNG CUỐI ---")
print(df.tail(10).to_string())
print("\nKiểu dữ liệu từng cột:")
print(df.dtypes)

# ══════════════════════════════════════════════════════════════════
# BƯỚC 2 – PHÂN LOẠI DỮ LIỆU & THANG ĐO
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BƯỚC 2 – PHÂN LOẠI DỮ LIỆU & THANG ĐO")
print("=" * 70)

cat_cols = df.select_dtypes(include=['object']).columns.tolist()
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nĐịnh TÍNH  (String) : {cat_cols}")
print(f"Định LƯỢNG (Số)     : {num_cols}")
print("""
Thang đo:
  STT         → Định danh (Nominal)  – số thứ tự, không có thứ bậc
  GT          → Định danh (Nominal)  – F/M
  DT          → Định danh (Nominal)  – dân tộc
  KV          → Thứ bậc   (Ordinal)  – khu vực ưu tiên 1 > 2 > 2NT
  KT          → Định danh (Nominal)  – khối thi A/B/C/D
  T1..N6      → Tỉ lệ     (Ratio)   – điểm học bạ 0–10
  DH1/DH2/DH3 → Tỉ lệ     (Ratio)   – điểm thi ĐH 0–10
""")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 3 – THỐNG KÊ & VISUALIZE DỮ LIỆU THIẾU
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("BƯỚC 3 – THỐNG KÊ DỮ LIỆU THIẾU + HEATMAP")
print("=" * 70)

miss     = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
miss_df  = pd.DataFrame({'Thiếu': miss, 'Tỉ lệ (%)': miss_pct})
miss_df  = miss_df[miss_df['Thiếu'] > 0].sort_values('Tỉ lệ (%)', ascending=False)
print(miss_df if len(miss_df) > 0 else "Không có dữ liệu thiếu!")

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=ax)
ax.set_title("Heatmap – Dữ liệu thiếu (TRƯỚC xử lý)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("proc_b3_heatmap_truoc.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════
# BƯỚC 4 – XỬ LÝ DỮ LIỆU THIẾU
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BƯỚC 4 – XỬ LÝ DỮ LIỆU THIẾU")
print("=" * 70)

# DT thiếu → 0
if 'DT' in df.columns:
    n_dt = df['DT'].isna().sum()
    # Bảng tần số trước
    freq = df['DT'].value_counts(dropna=False).reset_index()
    freq.columns = ['Giá trị', 'Tần số']
    freq['Tần suất (%)'] = (freq['Tần số'] / len(df) * 100).round(2)
    print("Bảng tần số DT (trước):")
    print(freq.to_string(index=False))
    df['DT'] = df['DT'].fillna(0)
    print(f"\nDT: {n_dt} dòng thiếu → điền 0 | còn thiếu: {df['DT'].isna().sum()}")

# Cột điểm thiếu → điền mean
score_cols = [c for c in num_cols if c not in ['STT', 'DT']]
print(f"\n{'Cột':<6} {'Thiếu':>7} {'Mean':>9} {'Sau':>6}")
print("-" * 28)
for col in score_cols:
    n = df[col].isna().sum()
    if n > 0:
        m = df[col].mean()
        df[col] = df[col].fillna(m)
        print(f"{col:<6} {n:>7} {m:>9.4f} {df[col].isna().sum():>6}")

# Cột string thiếu → mode
for col in cat_cols:
    n = df[col].isna().sum()
    if n > 0:
        m = df[col].mode()[0]
        df[col].fillna(m, inplace=True)
        print(f"[{col}] String: {n} thiếu → mode='{m}'")

print(f"\nTổng thiếu sau xử lý: {df.isnull().sum().sum()}")

fig, ax = plt.subplots(figsize=(16, 5))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=ax)
ax.set_title("Heatmap – Dữ liệu thiếu (SAU xử lý)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("proc_b4_heatmap_sau.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════
# BƯỚC 5 – CHUYỂN STRING → SỐ
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BƯỚC 5 – CHUYỂN STRING → SỐ")
print("=" * 70)

if 'GT' in df.columns:
    df['GT_num'] = df['GT'].map({'F': 0, 'M': 1})
    print(f"GT → GT_num: F=0, M=1 | {df['GT_num'].value_counts().to_dict()}")

if 'KV' in df.columns:
    kv_map = {v: i for i, v in enumerate(sorted(df['KV'].astype(str).unique()))}
    df['KV_num'] = df['KV'].astype(str).map(kv_map)
    print(f"KV → KV_num: {kv_map}")

if 'KT' in df.columns:
    kt_map = {v: i for i, v in enumerate(sorted(df['KT'].astype(str).unique()))}
    df['KT_num'] = df['KT'].astype(str).map(kt_map)
    print(f"KT → KT_num: {kt_map}")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 6 – PHÁT HIỆN & XỬ LÝ OUTLIER (IQR)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BƯỚC 6 – OUTLIER (IQR Method)")
print("=" * 70)

print("Ví dụ thầy: [5, 4, 7, 6, 3, 15, 40]")
demo = pd.Series([5, 4, 7, 6, 3, 15, 40])
Q1d, Q3d = demo.quantile(0.25), demo.quantile(0.75)
IQRd     = Q3d - Q1d
print(f"  Q1={Q1d}, Q3={Q3d}, IQR={IQRd}")
print(f"  Lower={Q1d-1.5*IQRd}  |  Upper={Q3d+1.5*IQRd}")
print(f"  Outlier: {demo[demo > Q3d+1.5*IQRd].tolist()}\n")

print(f"{'Cột':<6} {'Q1':>6} {'Q3':>6} {'IQR':>6} {'Lower':>7} {'Upper':>7} {'N_out':>7}")
print("-" * 50)
bounds = {}
for col in score_cols:
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    bounds[col] = (lo, hi)
    print(f"{col:<6} {Q1:>6.2f} {Q3:>6.2f} {IQR:>6.2f} {lo:>7.2f} {hi:>7.2f} {n_out:>7}")

# Boxplot
ncols_p = 4
nrows_p = (len(score_cols) + ncols_p - 1) // ncols_p
fig, axes = plt.subplots(nrows_p, ncols_p, figsize=(16, 4*nrows_p))
axes = axes.flatten()
for i, col in enumerate(score_cols):
    axes[i].boxplot(df[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    flierprops=dict(marker='o', color='red',
                                    markerfacecolor='red', markersize=4))
    axes[i].set_title(col, fontsize=9)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Boxplot – Outlier Detection (IQR)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("proc_b6_outlier_boxplot.png", dpi=150)
plt.show()

# Capping
for col in score_cols:
    lo, hi = bounds[col]
    n = ((df[col] < lo) | (df[col] > hi)).sum()
    df[col] = df[col].clip(lower=lo, upper=hi)
    if n > 0:
        print(f"  Capped '{col}': {n} outlier")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 7 – TẠO TBM1, TBM2, TBM3
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BƯỚC 7 – TẠO TBM (Trung Bình Môn)")
print("=" * 70)

def tinh_tbm(df, t, l, h, s, v, x, d, n):
    return (df[t]*2 + df[l] + df[h] + df[s] + df[v]*2 + df[x] + df[d] + df[n]) / 10

df['TBM1'] = tinh_tbm(df,'T1','L1','H1','S1','V1','X1','D1','N1')
df['TBM2'] = tinh_tbm(df,'T3','L3','H3','S3','V3','X3','D3','N3')
df['TBM3'] = tinh_tbm(df,'T5','L5','H5','S5','V5','X5','D5','N5')

print(df[['TBM1','TBM2','TBM3']].describe().round(3).to_string())

# ══════════════════════════════════════════════════════════════════
# BƯỚC 8 – XẾP LOẠI XL1, XL2, XL3
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BƯỚC 8 – XẾP LOẠI (XL)")
print("=" * 70)

def xep_loai(tbm):
    if   tbm < 5.0: return 'Y'
    elif tbm < 6.5: return 'TB'
    elif tbm < 8.0: return 'K'
    elif tbm < 9.0: return 'G'
    else:           return 'XS'

xl_map = {'Y':0,'TB':1,'K':2,'G':3,'XS':4}
for xl, tbm in [('XL1','TBM1'),('XL2','TBM2'),('XL3','TBM3')]:
    df[xl]        = df[tbm].apply(xep_loai)
    df[xl+'_num'] = df[xl].map(xl_map)
    print(f"\n{xl}: {df[xl].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 9 – MIN-MAX NORMALIZATION: VN(0-10) → US(0-4)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BƯỚC 9 – MIN-MAX: VN(0-10) → US(0-4)")
print("=" * 70)

for us_col, tbm_col in [('US_TBM1','TBM1'),('US_TBM2','TBM2'),('US_TBM3','TBM3')]:
    mn, mx = df[tbm_col].min(), df[tbm_col].max()
    df[us_col] = ((df[tbm_col] - mn) / (mx - mn) * 4).round(4)
    print(f"{us_col}: [{mn:.2f},{mx:.2f}] → US[{df[us_col].min():.4f},{df[us_col].max():.4f}]")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 10 – TẠO TARGET KQXT
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BƯỚC 10 – TẠO TARGET KQXT")
print("=" * 70)

def tinh_kqxt(row):
    kt = str(row['KT']).strip().upper()
    d1, d2, d3 = row['DH1'], row['DH2'], row['DH3']
    if   kt in ['A','A1']: diem = (d1*2 + d2 + d3) / 4
    elif kt == 'B':        diem = (d1 + d2*2 + d3) / 4
    else:                  diem = (d1 + d2 + d3) / 3
    return 1 if diem >= 5.0 else 0

df['KQXT'] = df.apply(tinh_kqxt, axis=1)
n_dau = df['KQXT'].sum()
n_rot = len(df) - n_dau
print(f"Đậu (1): {n_dau} ({n_dau/len(df)*100:.1f}%)  |  Rớt (0): {n_rot} ({n_rot/len(df)*100:.1f}%)")
print("\nTheo khối thi:")
print(df.groupby('KT')['KQXT'].value_counts().unstack(fill_value=0).to_string())

# ══════════════════════════════════════════════════════════════════
# BƯỚC 11 – INPUT (X) & TARGET (y)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BƯỚC 11 – INPUT (X) & TARGET (y)")
print("=" * 70)

target = 'KQXT'
input_features = [c for c in [
    'GT_num','KV_num','KT_num',
    'TBM1','TBM2','TBM3',
    'XL1_num','XL2_num','XL3_num',
    'US_TBM1','US_TBM2','US_TBM3',
    'DH1','DH2','DH3',
] if c in df.columns]

X = df[input_features]
y = df[target]
print(f"Target (y): '{target}'  |  X shape: {X.shape}")
print(f"Features  : {input_features}")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(df[input_features + [target]].corr(),
            annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, square=True, ax=ax)
ax.set_title("Correlation Matrix – Features & KQXT",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("proc_b11_correlation.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════
# BƯỚC 12 – LƯU FILE
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BƯỚC 12 – LƯU FILE")
print("=" * 70)

df.to_csv(OUT_PATH, index=False)
print(f"Đã lưu: {OUT_PATH}")
print(f"Kích thước cuối: {df.shape[0]} dòng  x  {df.shape[1]} cột")

print("\n✅ ProcessData HOÀN THÀNH!")
print("Dữ liệu sẵn sàng cho: Linear Regression / ANN / LSTM / MDR!")