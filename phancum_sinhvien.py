"""
PHÂN CỤM SINH VIÊN K58KTP THEO NĂNG LỰC HỌC TẬP
Thuật toán : K-Means (K=3)
Ngưỡng xếp loại:
    GPA >= 3.2  →  Giỏi
    GPA >= 2.5  →  Khá
    GPA <  2.5  →  Trung Bình
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════
# BƯỚC 1: ĐỌC VÀ LÀM SẠCH DỮ LIỆU
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("  PHÂN CỤM SINH VIÊN K58KTP  —  K-Means (K=3)")
print("  Ngưỡng: Giỏi ≥ 3.2 | Khá ≥ 2.5 | Trung Bình < 2.5")
print("=" * 60)

FILE = 'TỔNG HỢP ĐIỂM K58KTP.xlsx'
df = pd.read_excel(FILE, header=None)

student_ids   = df.iloc[1, 3:].tolist()
student_names = df.iloc[2, 3:].tolist()
subjects      = df.iloc[4:, 2].tolist()
raw_scores    = df.iloc[4:, 3:]

# Điểm bị Excel đọc sai thành ngày (VD: "3/7" → 3-Jul-2026)
DATE_MAP = {
    datetime.datetime(2026, 7, 3): 3.7,
    datetime.datetime(2026, 5, 3): 3.5,
    datetime.datetime(2026, 5, 2): 2.5,
    datetime.datetime(2026, 5, 1): 1.5,
    datetime.datetime(2026, 3, 2): 3.2,
}

def clean_score(v):
    if isinstance(v, datetime.datetime):
        return DATE_MAP.get(v, np.nan)
    if isinstance(v, str):
        s = v.strip()
        if s in ('/', '', '-'):
            return np.nan
        try:
            return float(s)
        except:
            return np.nan
    if isinstance(v, (int, float)):
        f = float(v)
        return np.nan if np.isnan(f) else f
    return np.nan

# Ma trận điểm: (71 sinh viên × 52 môn)
score_matrix = np.array([
    [clean_score(raw_scores.iloc[j, i]) for j in range(len(subjects))]
    for i in range(len(student_ids))
])
print(f"\n[✓] Đọc file: {len(student_ids)} sinh viên | {len(subjects)} môn học")

# ═══════════════════════════════════════════════════════════════
# BƯỚC 2: TÍNH ĐẶC TRƯNG (FEATURES)
# ═══════════════════════════════════════════════════════════════

def pct_in(arr, lo, hi):
    valid = arr[~np.isnan(arr)]
    return float(np.sum((valid >= lo) & (valid <= hi)) / len(valid)) if len(valid) else 0.0

records = []
for i in range(len(student_ids)):
    row   = score_matrix[i]
    valid = row[~np.isnan(row)]
    gpa   = float(np.nanmean(row)) if len(valid) > 0 else np.nan
    records.append({
        'MSSV'          : student_ids[i],
        'Ho_Ten'        : student_names[i],
        'GPA'           : gpa,
        'Ty_le_A'       : pct_in(row, 3.6, 4.0),
        'Ty_le_B'       : pct_in(row, 3.0, 3.59),
        'Ty_le_C'       : pct_in(row, 2.0, 2.99),
        'Ty_le_D'       : pct_in(row, 0.0, 1.99),
        'Do_lech_chuan' : float(np.nanstd(row)) if len(valid) > 1 else 0.0,
        'So_mon'        : int(np.sum(~np.isnan(row))),
    })

feat = pd.DataFrame(records)
median_gpa = feat['GPA'].median()

# 5 SV thiếu quá nhiều dữ liệu → GPA = median, xếp Khá
feat['GPA_goc'] = feat['GPA']           # lưu GPA gốc để ghi chú
feat['GPA']     = feat['GPA'].fillna(median_gpa)
print(f"[✓] GPA: {feat['GPA'].min():.3f} → {feat['GPA'].max():.3f}  (median = {median_gpa:.3f})")

# ═══════════════════════════════════════════════════════════════
# BƯỚC 3: XẾP LOẠI THEO NGƯỠNG GPA
# ═══════════════════════════════════════════════════════════════

def xep_loai(gpa):
    if gpa >= 3.2:
        return 'Giỏi'
    elif gpa >= 2.5:
        return 'Khá'
    else:
        return 'Trung Bình'

feat['Nang_luc'] = feat['GPA'].apply(xep_loai)

ORDER  = ['Giỏi', 'Khá', 'Trung Bình']
COLORS = {'Giỏi': '#27ae60', 'Khá': '#2980b9', 'Trung Bình': '#e67e22'}
MARKS  = {'Giỏi': 'o',       'Khá': 's',       'Trung Bình': '^'}

# ═══════════════════════════════════════════════════════════════
# BƯỚC 4: CHẠY K-MEANS ĐỂ PHÂN TÍCH CỤM (K=3)
# ═══════════════════════════════════════════════════════════════

feature_cols = ['GPA', 'Ty_le_A', 'Ty_le_B', 'Ty_le_C', 'Do_lech_chuan']
X = feat[feature_cols].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=20, max_iter=500)
feat['Cluster'] = kmeans.fit_predict(X_scaled)
sil = silhouette_score(X_scaled, feat['Cluster'])
print(f"[✓] K-Means (K=3) | Silhouette Score: {sil:.4f}")

# ═══════════════════════════════════════════════════════════════
# BƯỚC 5: IN KẾT QUẢ
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  KẾT QUẢ PHÂN LOẠI")
print("=" * 60)

for nl in ORDER:
    grp = feat[feat['Nang_luc'] == nl].sort_values('GPA', ascending=False)
    print(f"\n┌─ {nl.upper()} ({len(grp)} sinh viên) "
          f"| GPA TB: {grp['GPA'].mean():.3f} "
          f"| [{grp['GPA'].min():.3f} – {grp['GPA'].max():.3f}]")
    for _, r in grp.iterrows():
        flag = '  ⚠ thiếu dữ liệu' if pd.isna(r['GPA_goc']) else ''
        print(f"│  {r['MSSV']}  {r['Ho_Ten']:<28}  GPA: {r['GPA']:.3f}{flag}")
    print("└" + "─" * 55)

# ═══════════════════════════════════════════════════════════════
# BƯỚC 6: BIỂU ĐỒ 1 — PHÂN BỐ GPA + ĐƯỜNG NGƯỠNG
# ═══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(13, 5))
ax.set_facecolor('#f8f9fa')

sv_sorted = feat.sort_values('GPA', ascending=False).reset_index(drop=True)
bar_colors = [COLORS[nl] for nl in sv_sorted['Nang_luc']]

bars = ax.bar(range(len(sv_sorted)), sv_sorted['GPA'],
              color=bar_colors, edgecolor='white', linewidth=0.4, width=0.85)

# Đường ngưỡng
ax.axhline(y=3.2, color='#27ae60', linestyle='--', linewidth=1.8, label='Ngưỡng Giỏi (3.2)')
ax.axhline(y=2.5, color='#2980b9', linestyle='--', linewidth=1.8, label='Ngưỡng Khá (2.5)')

# Vùng màu nền
ax.axhspan(3.2, 4.0, alpha=0.06, color='#27ae60')
ax.axhspan(2.5, 3.2, alpha=0.06, color='#2980b9')
ax.axhspan(0.0, 2.5, alpha=0.06, color='#e67e22')

# Nhãn chú thích vùng
ax.text(len(sv_sorted) * 0.98, 3.6,  'GIỎI',      ha='right', fontsize=10, color='#27ae60', fontweight='bold', alpha=0.7)
ax.text(len(sv_sorted) * 0.98, 2.85, 'KHÁ',       ha='right', fontsize=10, color='#2980b9', fontweight='bold', alpha=0.7)
ax.text(len(sv_sorted) * 0.98, 2.15, 'TRUNG BÌNH', ha='right', fontsize=10, color='#e67e22', fontweight='bold', alpha=0.7)

ax.set_xticks(range(len(sv_sorted)))
ax.set_xticklabels([n.split()[-1] for n in sv_sorted['Ho_Ten']],
                   rotation=75, fontsize=7, ha='right')
ax.set_ylabel('GPA (thang 4.0)', fontsize=11)
ax.set_title('Phân bố GPA sinh viên K58KTP theo Ngưỡng Năng lực', fontsize=13, fontweight='bold')
ax.set_ylim(0, 4.15)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, axis='y', alpha=0.25, linestyle='--')

legend_patches = [mpatches.Patch(color=COLORS[nl], label=f'{nl} ({len(feat[feat["Nang_luc"]==nl])} SV)')
                  for nl in ORDER]
ax.legend(handles=legend_patches + [
    plt.Line2D([0],[0], color='#27ae60', linestyle='--', label='Ngưỡng 3.2'),
    plt.Line2D([0],[0], color='#2980b9', linestyle='--', label='Ngưỡng 2.5'),
], fontsize=9, loc='upper right', ncol=2)

plt.tight_layout()
plt.savefig('bieu_do_1_gpa_nguong.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[✓] Đã lưu: bieu_do_1_gpa_nguong.png")

# ═══════════════════════════════════════════════════════════════
# BƯỚC 7: BIỂU ĐỒ 2 — PCA 2D PHÂN CỤM
# ═══════════════════════════════════════════════════════════════

pca  = PCA(n_components=2, random_state=42)
Xpca = pca.fit_transform(X_scaled)
var1, var2 = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_facecolor('#f8f9fa')

for nl in ORDER:
    idx = feat[feat['Nang_luc'] == nl].index.tolist()
    ax.scatter(Xpca[idx, 0], Xpca[idx, 1],
               c=COLORS[nl], marker=MARKS[nl], s=140,
               label=nl, alpha=0.88, edgecolors='white', linewidth=0.8, zorder=3)
    for i in idx:
        ax.annotate(feat.loc[i, 'Ho_Ten'].split()[-1],
                    (Xpca[i, 0], Xpca[i, 1]),
                    xytext=(0, 7), textcoords='offset points',
                    fontsize=6.5, ha='center', color=COLORS[nl], fontweight='bold')

# Tâm cụm K-Means (★)
centers_pca = pca.transform(kmeans.cluster_centers_)
# Gán màu tâm theo xếp loại GPA của centroid
c_gpas = scaler.inverse_transform(kmeans.cluster_centers_)[:, 0]
for ci, cgpa in enumerate(c_gpas):
    cn = xep_loai(cgpa)
    ax.scatter(centers_pca[ci, 0], centers_pca[ci, 1],
               c=COLORS[cn], s=380, marker='*',
               edgecolors='black', linewidth=1.5, zorder=5,
               label=f'Tâm cụm ({cn})' if ci == 0 else '')

# Đường biên ngưỡng chú thích
for nl in ORDER:
    grp = feat[feat['Nang_luc'] == nl]
    if len(grp) == 0: continue
    idx = grp.index.tolist()
    hull_x = [Xpca[i, 0] for i in idx]
    hull_y = [Xpca[i, 1] for i in idx]
    cx, cy = np.mean(hull_x), np.mean(hull_y)
    ax.text(cx, cy, nl, ha='center', va='center',
            fontsize=14, color=COLORS[nl], fontweight='bold', alpha=0.25)

ax.set_xlabel(f'Thành phần chính 1 ({var1:.1f}% phương sai)', fontsize=11)
ax.set_ylabel(f'Thành phần chính 2 ({var2:.1f}% phương sai)', fontsize=11)
ax.set_title(
    'Phân cụm K-Means (K=3) sinh viên K58KTP\n'
    f'Ngưỡng: Giỏi ≥ 3.2 | Khá ≥ 2.5 | Trung Bình < 2.5  '
    f'(Silhouette = {sil:.4f})',
    fontsize=12, fontweight='bold')

legend_p = [mpatches.Patch(color=COLORS[nl],
            label=f'{nl}  ({len(feat[feat["Nang_luc"]==nl])} SV | '
                  f'GPA TB {feat[feat["Nang_luc"]==nl]["GPA"].mean():.3f})')
            for nl in ORDER]
ax.legend(handles=legend_p, fontsize=10, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--')
plt.tight_layout()
plt.savefig('bieu_do_2_pca_cluster.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Đã lưu: bieu_do_2_pca_cluster.png")

# ═══════════════════════════════════════════════════════════════
# BƯỚC 8: BIỂU ĐỒ 3 — BOXPLOT GPA
# ═══════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 6))
ax.set_facecolor('#f8f9fa')

box_data = [feat[feat['Nang_luc'] == nl]['GPA'].values for nl in ORDER]
bp = ax.boxplot(box_data, patch_artist=True, notch=False,
                medianprops=dict(color='black', linewidth=2.2),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5))
for patch, nl in zip(bp['boxes'], ORDER):
    patch.set_facecolor(COLORS[nl]); patch.set_alpha(0.72)

for i, nl in enumerate(ORDER):
    y = feat[feat['Nang_luc'] == nl]['GPA'].values
    ax.scatter(np.random.normal(i+1, 0.07, len(y)), y,
               c=COLORS[nl], s=50, alpha=0.75, edgecolors='white', linewidth=0.5, zorder=3)
    ax.text(i+1, feat[feat['Nang_luc']==nl]['GPA'].mean(),
            f"TB:{feat[feat['Nang_luc']==nl]['GPA'].mean():.3f}",
            ha='center', va='bottom', fontsize=8.5, color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

# Vẽ đường ngưỡng
ax.axhline(3.2, linestyle='--', color='#27ae60', linewidth=1.5, label='Ngưỡng Giỏi (3.2)')
ax.axhline(2.5, linestyle='--', color='#2980b9', linewidth=1.5, label='Ngưỡng Khá (2.5)')

ax.set_xticks([1,2,3])
ax.set_xticklabels(
    [f'{nl}\n(n={len(feat[feat["Nang_luc"]==nl])})' for nl in ORDER],
    fontsize=11, fontweight='bold')
ax.set_ylabel('GPA (thang 4.0)', fontsize=12)
ax.set_title('Phân phối GPA theo Cụm Năng lực', fontsize=13, fontweight='bold')
ax.set_ylim(1.3, 4.1)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('bieu_do_3_boxplot_gpa.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Đã lưu: bieu_do_3_boxplot_gpa.png")

# ═══════════════════════════════════════════════════════════════
# BƯỚC 9: BIỂU ĐỒ 4 — PIE CHART TỈ LỆ
# ═══════════════════════════════════════════════════════════════

counts = [len(feat[feat['Nang_luc'] == nl]) for nl in ORDER]
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Pie chart
wedges, texts, autotexts = axes[0].pie(
    counts,
    labels=[f'{nl}\n({c} SV)' for nl, c in zip(ORDER, counts)],
    colors=[COLORS[nl] for nl in ORDER],
    autopct='%1.1f%%',
    startangle=140,
    pctdistance=0.65,
    wedgeprops=dict(edgecolor='white', linewidth=2))
for at in autotexts:
    at.set_fontsize(12); at.set_fontweight('bold')
axes[0].set_title('Tỉ lệ sinh viên theo Năng lực', fontsize=12, fontweight='bold')

# Horizontal bar: đặc trưng trung bình từng cụm
feat_means = feat.groupby('Nang_luc')[['GPA','Ty_le_A','Ty_le_B','Ty_le_C']].mean()
bar_labels  = ['GPA (÷4)', 'Tỉ lệ Điểm A', 'Tỉ lệ Điểm B', 'Tỉ lệ Điểm C']
y = np.arange(len(bar_labels))
width = 0.25

for k, nl in enumerate(ORDER):
    vals = [
        feat_means.loc[nl, 'GPA'] / 4,
        feat_means.loc[nl, 'Ty_le_A'],
        feat_means.loc[nl, 'Ty_le_B'],
        feat_means.loc[nl, 'Ty_le_C'],
    ]
    axes[1].barh(y + k*width, vals, width, label=nl, color=COLORS[nl], alpha=0.82, edgecolor='white')

axes[1].set_yticks(y + width)
axes[1].set_yticklabels(bar_labels, fontsize=10)
axes[1].set_xlabel('Giá trị trung bình (chuẩn hóa 0–1)', fontsize=10)
axes[1].set_title('So sánh đặc trưng từng Cụm', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, axis='x', alpha=0.3, linestyle='--')
axes[1].set_facecolor('#f8f9fa')

plt.suptitle('Tổng hợp Phân tích Năng lực Sinh viên K58KTP',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('bieu_do_4_tong_hop.png', dpi=150, bbox_inches='tight')
plt.close()
print("[✓] Đã lưu: bieu_do_4_tong_hop.png")

# ═══════════════════════════════════════════════════════════════
# BƯỚC 10: XUẤT EXCEL KẾT QUẢ
# ═══════════════════════════════════════════════════════════════

out = feat[['MSSV','Ho_Ten','GPA','Ty_le_A','Ty_le_B','Ty_le_C','Ty_le_D',
            'Do_lech_chuan','So_mon','Nang_luc']].copy()
out.columns = ['MSSV','Họ Tên','GPA','Tỉ lệ A (≥3.6)','Tỉ lệ B (3.0-3.5)',
               'Tỉ lệ C (2.0-2.9)','Tỉ lệ D (<2.0)',
               'Độ lệch chuẩn','Số môn','Năng lực']
out = out.sort_values(['Năng lực','GPA'], ascending=[True,False]).reset_index(drop=True)
out.index += 1

with pd.ExcelWriter('ket_qua_phan_cum_3cum.xlsx', engine='openpyxl') as writer:
    out.to_excel(writer, sheet_name='Kết quả phân cụm')
    # Sheet thống kê
    stats = feat.groupby('Nang_luc').agg(
        So_SV=('GPA','count'),
        GPA_TB=('GPA','mean'),
        GPA_Min=('GPA','min'),
        GPA_Max=('GPA','max'),
        GPA_Std=('GPA','std'),
        TyLe_A_TB=('Ty_le_A','mean'),
        TyLe_B_TB=('Ty_le_B','mean'),
        TyLe_C_TB=('Ty_le_C','mean'),
    ).round(4)
    stats.to_excel(writer, sheet_name='Thống kê theo cụm')

print("[✓] Đã lưu: ket_qua_phan_cum_3cum.xlsx")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TỔNG KẾT")
print("=" * 60)
print(f"  {'Cụm':<14} {'Số SV':>6}  {'GPA TB':>8}  {'GPA Min':>8}  {'GPA Max':>8}")
print(f"  {'-'*52}")
for nl in ORDER:
    g = feat[feat['Nang_luc']==nl]
    print(f"  {nl:<14} {len(g):>6}  {g['GPA'].mean():>8.3f}  {g['GPA'].min():>8.3f}  {g['GPA'].max():>8.3f}")
print(f"\n  Silhouette Score (K-Means) = {sil:.4f}")
print(f"  Ngưỡng phân loại: Giỏi ≥ 3.2 | Khá ≥ 2.5 | TB < 2.5")
print("=" * 60)
print("\n  Files đầu ra:")
for f in ['bieu_do_1_gpa_nguong.png','bieu_do_2_pca_cluster.png',
          'bieu_do_3_boxplot_gpa.png','bieu_do_4_tong_hop.png',
          'ket_qua_phan_cum_3cum.xlsx']:
    print(f"    {f}")
print("\n  HOÀN THÀNH!")