"""
╔══════════════════════════════════════════════════════════════════╗
║  LAB 04 –    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120

DATA_PATH = r"E:\MonDeepLearning\DataAnalystDeepLearning-main\DataAnalystDeepLearning-main\Data\titanic_disaster.csv"

# ══════════════════════════════════════════════════════════════════
# BƯỚC 1 – LOAD DATA
# ══════════════════════════════════════════════════════════════════
def load_data(path):
    df = pd.read_csv(path)
    print("=" * 65)
    print("BƯỚC 1 – LOAD DATA")
    print("=" * 65)
    print(f"Kích thước: {df.shape[0]} dòng  x  {df.shape[1]} cột")
    print("\n10 dòng đầu:")
    print(df.head(10).to_string())
    print("\nKiểu dữ liệu từng cột:")
    print(df.dtypes)
    return df

df = load_data(DATA_PATH)

# ══════════════════════════════════════════════════════════════════
# BƯỚC 2 – THỐNG KÊ & VISUALIZE DỮ LIỆU THIẾU (Heatmap)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 2 – THỐNG KÊ DỮ LIỆU THIẾU + HEATMAP")
print("=" * 65)

miss     = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
miss_df  = pd.DataFrame({'Thiếu': miss, 'Tỉ lệ (%)': miss_pct})
miss_df  = miss_df[miss_df['Thiếu'] > 0].sort_values('Tỉ lệ (%)', ascending=False)
print(miss_df)
print("""
Nhận xét:
  • Cabin    ~77%  → quá nhiều, gán 'Unknown'
  • Age      ~20%  → điền median theo nhóm Pclass
  • Embarked   2   → điền mode
""")

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=ax)
ax.set_title("Heatmap – Dữ liệu thiếu (TRƯỚC xử lý)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("b2_heatmap_truoc.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════
# BƯỚC 3 – XỬ LÝ DỮ LIỆU THIẾU
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 3 – XỬ LÝ DỮ LIỆU THIẾU")
print("=" * 65)

# 3a. Cabin → typeCabin (lấy chữ cái đầu, thiếu = 'Unknown')
df['typeCabin'] = df['Cabin'].apply(
    lambda x: str(x)[0] if pd.notna(x) else 'Unknown')
df.drop(columns=['Cabin'], inplace=True)
print("3a. typeCabin:", df['typeCabin'].value_counts().to_dict())

# 3b. Embarked → điền mode
mode_emb = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_emb, inplace=True)
print(f"3b. Embarked – điền mode='{mode_emb}', còn thiếu: {df['Embarked'].isna().sum()}")

# 3c. Age → boxplot theo Pclass → điền median theo nhóm
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x='Pclass', y='Age', data=df, palette='Set2', ax=ax)
ax.set_title("Boxplot Age theo Pclass (trước khi fill)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("b3_boxplot_age_pclass.png", dpi=150)
plt.show()

print("\nMedian Age theo từng Pclass:")
med_age = df.groupby('Pclass')['Age'].median()
print(med_age.round(2))
print("→ Tuổi khác nhau theo hạng vé → dùng median từng nhóm Pclass")

df['Age'] = df.apply(
    lambda r: med_age[r['Pclass']] if pd.isnull(r['Age']) else r['Age'], axis=1)
print(f"Age còn thiếu sau xử lý: {df['Age'].isna().sum()}")

# Heatmap SAU xử lý
fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=ax)
ax.set_title("Heatmap – Dữ liệu thiếu (SAU xử lý)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("b3_heatmap_sau.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════
# BƯỚC 4 – CHUYỂN STRING → SỐ (Encode Categorical)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 4 – CHUYỂN STRING → SỐ")
print("=" * 65)

# 4a. Tách Name → secondName + firstName
df['secondName'] = df['Name'].apply(lambda x: x.split(',')[0].strip())
df['firstName']  = df['Name'].apply(
    lambda x: x.split(',')[1].strip() if ',' in x else '')
df.drop(columns=['Name'], inplace=True)
print("4a. Đã tách Name → secondName, firstName")

# 4b. namePrefix từ firstName → mã số
def extract_prefix(name):
    for p in ['Mr.', 'Mrs.', 'Miss.', 'Master.']:
        if p in name: return p.replace('.', '')
    return 'Other'

df['namePrefix']     = df['firstName'].apply(extract_prefix)
prefix_map           = {'Mr': 1, 'Mrs': 2, 'Miss': 3, 'Master': 4, 'Other': 0}
df['namePrefix_num'] = df['namePrefix'].map(prefix_map)
print("4b. namePrefix:", df['namePrefix'].value_counts().to_dict())

# 4c. Sex: male=1, female=0
df['Sex_num'] = df['Sex'].map({'male': 1, 'female': 0})
print("4c. Sex_num:", df['Sex_num'].value_counts().to_dict())

# 4d. Embarked: S=0, C=1, Q=2
df['Embarked_num'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
print("4d. Embarked_num:", df['Embarked_num'].value_counts().to_dict())

# 4e. typeCabin → số
cabin_map            = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'T':8,'Unknown':0}
df['typeCabin_num']  = df['typeCabin'].map(cabin_map).fillna(0).astype(int)
print("4e. typeCabin_num:", df['typeCabin_num'].value_counts().sort_index().to_dict())

# ══════════════════════════════════════════════════════════════════
# BƯỚC 5 – FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 5 – FEATURE ENGINEERING")
print("=" * 65)

# AgeGroup
def age_group(age):
    if age <= 12:   return 'Kid'
    elif age <= 18: return 'Teen'
    elif age <= 60: return 'Adult'
    else:           return 'Older'

df['AgeGroup']     = df['Age'].apply(age_group)
df['AgeGroup_num'] = df['AgeGroup'].map({'Kid':0,'Teen':1,'Adult':2,'Older':3})
print("5a. AgeGroup:", df['AgeGroup'].value_counts().to_dict())

# familySize & Alone
df['familySize'] = 1 + df['SibSp'] + df['Parch']
df['Alone']      = (df['familySize'] == 1).astype(int)
print("5b. familySize range:", df['familySize'].min(), "–", df['familySize'].max())
print("5c. Alone:", df['Alone'].value_counts().to_dict())

# ══════════════════════════════════════════════════════════════════
# BƯỚC 6 – PHÁT HIỆN & XỬ LÝ OUTLIER (IQR)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 6 – OUTLIER (IQR Method)")
print("=" * 65)

# Minh họa IQR với ví dụ của thầy
print("Ví dụ thầy: [5, 4, 7, 6, 3, 15, 40]")
demo = pd.Series([5, 4, 7, 6, 3, 15, 40])
Q1d, Q3d, IQRd = demo.quantile(0.25), demo.quantile(0.75), demo.quantile(0.75)-demo.quantile(0.25)
print(f"  Q1={Q1d}, Q3={Q3d}, IQR={IQRd}")
print(f"  Lower = {Q1d - 1.5*IQRd}  |  Upper = {Q3d + 1.5*IQRd}")
print(f"  Outlier: {demo[demo > Q3d + 1.5*IQRd].tolist()}")

# Phát hiện outlier trên dữ liệu thực
num_features = ['Age', 'Fare', 'familySize']
print(f"\n{'Cột':<12} {'Q1':>7} {'Q3':>7} {'IQR':>7} {'Lower':>8} {'Upper':>8} {'Outlier':>8}")
print("-" * 63)
outlier_bounds = {}
for col in num_features:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    outlier_bounds[col] = (lower, upper)
    print(f"{col:<12} {Q1:>7.2f} {Q3:>7.2f} {IQR:>7.2f} {lower:>8.2f} {upper:>8.2f} {n_out:>8}")

# Boxplot visualize outlier
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Boxplot – Phát hiện Outlier", fontsize=14, fontweight='bold')
for i, col in enumerate(num_features):
    axes[i].boxplot(df[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    flierprops=dict(marker='o', color='red',
                                    markerfacecolor='red', markersize=5))
    axes[i].set_title(col)
plt.tight_layout()
plt.savefig("b6_outlier_boxplot.png", dpi=150)
plt.show()

# Xử lý: capping tại ngưỡng Lower/Upper
df_clean = df.copy()
for col in num_features:
    lo, hi = outlier_bounds[col]
    n = ((df_clean[col] < lo) | (df_clean[col] > hi)).sum()
    df_clean[col] = df_clean[col].clip(lower=lo, upper=hi)
    print(f"Capped '{col}': {n} outlier")

# ══════════════════════════════════════════════════════════════════
# BƯỚC 7 – XÁC ĐỊNH INPUT (X) & TARGET (y) cho Model
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 7 – INPUT (X) và TARGET (y)")
print("=" * 65)

input_features = [
    'Pclass','Sex_num','Age','Fare','familySize',
    'Alone','AgeGroup_num','namePrefix_num',
    'Embarked_num','typeCabin_num',
]
target = 'Survived'

X = df_clean[input_features]
y = df_clean[target]
print(f"Target (y) : '{target}'  →  {y.value_counts().to_dict()}")
print(f"Input  (X) : {X.shape[1]} features, {X.shape[0]} mẫu")

# Correlation heatmap
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(df_clean[input_features + [target]].corr(),
            annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, square=True, ax=ax)
ax.set_title("Correlation Matrix – Features & Target", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("b7_correlation_matrix.png", dpi=150)
plt.show()

# ══════════════════════════════════════════════════════════════════
# BƯỚC 8 – EDA
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("BƯỚC 8 – EDA")
print("=" * 65)

# Sống sót theo Sex
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Bài 12 – Sống Sót theo Giới Tính", fontsize=14, fontweight='bold')
sns.countplot(x='Sex', hue='Survived', data=df_clean, palette='Set1', ax=axes[0])
axes[0].legend(['Thiệt mạng','Sống sót'])
sr = df_clean.groupby('Sex')['Survived'].mean().reset_index()
sns.barplot(x='Sex', y='Survived', data=sr, palette='Set2', ax=axes[1])
axes[1].set_ylabel("Tỉ lệ sống sót")
axes[1].set_ylim(0, 1)
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.2%}',
                     (p.get_x()+p.get_width()/2, p.get_height()),
                     ha='center', va='bottom', fontsize=12)
plt.tight_layout(); plt.savefig("eda_b12_sex.png", dpi=150); plt.show()

# Sống sót theo Pclass
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Bài 13 – Sống Sót theo Hạng Vé", fontsize=14, fontweight='bold')
sns.countplot(x='Pclass', hue='Survived', data=df_clean, palette='coolwarm', ax=axes[0])
axes[0].legend(['Thiệt mạng','Sống sót'])
sr2 = df_clean.groupby('Pclass')['Survived'].mean().reset_index()
sns.barplot(x='Pclass', y='Survived', data=sr2, palette='Blues_d', ax=axes[1])
axes[1].set_ylabel("Tỉ lệ sống sót"); axes[1].set_ylim(0, 1)
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.2%}',
                     (p.get_x()+p.get_width()/2, p.get_height()),
                     ha='center', va='bottom', fontsize=12)
plt.tight_layout(); plt.savefig("eda_b13_pclass.png", dpi=150); plt.show()

# Sống sót theo AgeGroup & Sex
order = ['Kid','Teen','Adult','Older']
g = sns.FacetGrid(df_clean, col='Sex', height=5, aspect=1.2)
g.map_dataframe(sns.countplot, x='AgeGroup', hue='Survived', palette='Set1', order=order)
g.add_legend(title='Survived')
g.set_titles(col_template="Sex: {col_name}")
g.set_axis_labels("Nhóm tuổi", "Số lượng")
g.figure.suptitle("Bài 14 – AgeGroup & Sex", fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout(); plt.savefig("eda_b14_agegroup.png", dpi=150); plt.show()

# familySize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Bài 15 – Sống Sót theo familySize", fontsize=14, fontweight='bold')
sns.countplot(x='familySize', hue='Survived', data=df_clean, palette='Set2', ax=axes[0])
axes[0].legend(['Thiệt mạng','Sống sót'])
sr3 = df_clean.groupby('familySize')['Survived'].mean().reset_index()
sns.barplot(x='familySize', y='Survived', data=sr3, palette='Greens_d', ax=axes[1])
axes[1].set_ylim(0, 1)
for p in axes[1].patches:
    axes[1].annotate(f'{p.get_height():.2%}',
                     (p.get_x()+p.get_width()/2, p.get_height()),
                     ha='center', va='bottom', fontsize=9)
plt.tight_layout(); plt.savefig("eda_b15_familysize.png", dpi=150); plt.show()

# Fare
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Bài 16 – Sống Sót theo Fare", fontsize=14, fontweight='bold')
axes[0].hist([df_clean[df_clean['Survived']==0]['Fare'],
              df_clean[df_clean['Survived']==1]['Fare']],
             bins=30, label=['Thiệt mạng','Sống sót'],
             color=['#e74c3c','#2ecc71'], alpha=0.7)
axes[0].legend()
sns.kdeplot(data=df_clean, x='Fare', hue='Survived', fill=True,
            palette={0:'#e74c3c',1:'#2ecc71'}, ax=axes[1])
axes[1].set_xlim(0, 300)
plt.tight_layout(); plt.savefig("eda_b16_fare.png", dpi=150); plt.show()

# Pclass & Embarked heatmap
pivot = df_clean.pivot_table(values='Survived', index='Embarked',
                              columns='Pclass', aggfunc='mean')
fig, ax = plt.subplots(figsize=(8, 4))
sns.heatmap(pivot, annot=True, fmt='.2%', cmap='YlGnBu', linewidths=0.5, ax=ax)
ax.set_title("Bài 17 – Heatmap Sống Sót theo Pclass & Embarked",
             fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig("eda_b17_heatmap.png", dpi=150); plt.show()

# Displot + swarm + factorplot (demo như hình hướng dẫn)
sns.displot(df_clean, x="Age", col="Pclass", kde=True)
plt.suptitle("Phân phối Tuổi theo Pclass", y=1.02, fontsize=13, fontweight='bold')
plt.savefig("demo_displot.png", dpi=150, bbox_inches='tight'); plt.show()

sns.catplot(x="Pclass", y="Age", kind="swarm", hue="Sex", data=df_clean)
plt.suptitle("Swarm – Age / Pclass / Sex", y=1.02, fontsize=13, fontweight='bold')
plt.savefig("demo_swarm.png", dpi=150, bbox_inches='tight'); plt.show()

sns.catplot(x='Pclass', y='Survived', hue='Sex', col='Embarked',
            data=df_clean, kind='bar', palette='Set1')
plt.suptitle("Sống Sót theo Pclass, Sex & Embarked", y=1.02,
             fontsize=13, fontweight='bold')
plt.savefig("demo_factorplot.png", dpi=150, bbox_inches='tight'); plt.show()

print("\n✅ LAB 04 HOÀN THÀNH!")
print(f"X shape: {X.shape}  |  y shape: {y.shape}")
print("Dataset đã sẵn sàng cho Regression / Classification Model.")