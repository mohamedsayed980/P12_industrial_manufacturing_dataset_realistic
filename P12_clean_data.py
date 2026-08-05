"""
P12 Industrial Manufacturing — Data Cleaning & Feature Engineering
Run this in Jupyter ONCE → saves manufacturing_clean.csv
Mohamed · M3
Dataset: Industrial Manufacturing Realistic · 54,750 shift records · 2025
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ── 1. LOAD ───────────────────────────────────────────────────
df = pd.read_csv('industrial_manufacturing_dataset_realistic.csv')
print(f"✅ Raw shape     : {df.shape}")
print(f"   Columns       : {df.columns.tolist()}")

# ── 2. NULL ANALYSIS ─────────────────────────────────────────
print(f"\n✅ Total nulls   : {df.isnull().sum().sum()}")
print("\nNull breakdown:")
for col in df.columns:
    n = df[col].isnull().sum()
    if n > 0:
        print(f"  {col:20s}: {n:,} ({n/len(df)*100:.1f}%)")

# ── 3. PARSE DATE ─────────────────────────────────────────────
df['date'] = pd.to_datetime(df['date'], errors='coerce')
print(f"\n✅ Date range    : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"   Total days    : {df['date'].nunique()}")

# ── 4. FILL INFORMATIVE NULLS ─────────────────────────────────
# NaN = no downtime / no defect — these are INFORMATIVE, not missing
df['downtime_reason'] = df['downtime_reason'].fillna('No Downtime')
df['defect_type']     = df['defect_type'].fillna('No Defect')
print(f"\n✅ downtime_reason filled: {df['downtime_reason'].value_counts().to_dict()}")
print(f"\n✅ defect_type filled    : {df['defect_type'].value_counts().to_dict()}")
print(f"\n✅ Total nulls after fill: {df.isnull().sum().sum()}")

# ── 5. FEATURE ENGINEERING ───────────────────────────────────

# Time features
df['month']       = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek   # 0=Mon … 6=Sun
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
df['week']        = df['date'].dt.isocalendar().week.astype(int)
print(f"\n✅ Time features: month, day_of_week, is_weekend, week")

# Downtime rate
df['downtime_rate'] = (df['downtime_min'] /
                       df['planned_time_min'].replace(0, 1)).round(4)
print(f"✅ downtime_rate: mean={df['downtime_rate'].mean():.4f}")

# Reject rate (defect rate per shift)
df['reject_rate'] = (df['reject_units'] /
                     df['total_units'].replace(0, 1)).round(4)
print(f"✅ reject_rate  : mean={df['reject_rate'].mean():.4f} ({df['reject_rate'].mean()*100:.2f}%)")

# Energy per good unit produced
df['energy_per_unit'] = (df['energy_kwh'] /
                          df['good_units'].replace(0, 1)).round(4)
print(f"✅ energy_per_unit: mean={df['energy_per_unit'].mean():.4f} kWh/unit")

# OEE gap from world class (85%)
df['oee_gap'] = (0.85 - df['OEE']).round(4)
print(f"✅ oee_gap: mean={df['oee_gap'].mean():.4f} (positive = below world class)")

# No downtime flag
df['no_downtime'] = (df['downtime_min'] == 0).astype(int)
nd_oee = df.groupby('no_downtime')['OEE'].mean()
print(f"\n✅ no_downtime — OEE comparison:")
print(f"   With downtime   : {nd_oee.get(0,0):.4f}")
print(f"   Without downtime: {nd_oee.get(1,0):.4f}")

# Has breakdown flag
df['has_breakdown'] = (df['downtime_reason'] == 'Breakdown').astype(int)
bd_oee = df.groupby('has_breakdown')['OEE'].mean()
print(f"\n✅ has_breakdown — OEE comparison:")
print(f"   No breakdown  : {bd_oee.get(0,0):.4f}")
print(f"   Has breakdown : {bd_oee.get(1,0):.4f}")

# Has defect flag
df['has_defect'] = (df['defect_type'] != 'No Defect').astype(int)
hd_oee = df.groupby('has_defect')['OEE'].mean()
print(f"\n✅ has_defect — OEE comparison:")
print(f"   No defect  : {hd_oee.get(0,0):.4f}")
print(f"   Has defect : {hd_oee.get(1,0):.4f}")

# Production efficiency
df['production_efficiency'] = (df['good_units'] /
                                df['total_units'].replace(0, 1)).round(4)
print(f"\n✅ production_efficiency: mean={df['production_efficiency'].mean():.4f}")

# OEE Classification target
# low_OEE = 1 if OEE < 0.65 (poor performance threshold)
df['low_OEE'] = (df['OEE'] < 0.65).astype(int)
low_rate = df['low_OEE'].mean() * 100
print(f"\n✅ low_OEE target:")
print(f"   Low OEE (<0.65) : {df['low_OEE'].sum():,} ({low_rate:.1f}%)")
print(f"   Normal OEE      : {(df['low_OEE']==0).sum():,} ({100-low_rate:.1f}%)")
print("   → Moderate imbalance → class_weight='balanced' needed")

# OEE category
df['OEE_category'] = pd.cut(df['OEE'],
    bins=[0, 0.50, 0.65, 0.75, 0.85, 1.01],
    labels=['Critical(<50%)','Poor(50-65%)','Fair(65-75%)','Good(75-85%)','World Class(>85%)'])
print(f"\n✅ OEE_category distribution:")
print(df['OEE_category'].value_counts().sort_index())

# Label Encoding for categorical columns
le = LabelEncoder()
cat_cols = ['shift','machine_id','product_id','operator_id',
            'downtime_reason','defect_type']
for col in cat_cols:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    print(f"✅ {col}_enc: {df[col].nunique()} categories")

# ── 6. FINAL CHECK ───────────────────────────────────────────
print(f"\n✅ Final shape   : {df.shape}")
print(f"✅ Total nulls   : {df.isnull().sum().sum()}")
print(f"✅ Memory        : {df.memory_usage(deep=True).sum()/1024**2:.2f} MB")

print("\n=== OEE SUMMARY ===")
print(f"  Mean OEE        : {df['OEE'].mean():.4f} ({df['OEE'].mean()*100:.1f}%)")
print(f"  World class≥85% : {(df['OEE']>=0.85).mean()*100:.1f}% of shifts")
print(f"  Poor OEE <65%   : {(df['OEE']<0.65).mean()*100:.1f}% of shifts")

print("\n=== DOWNTIME SUMMARY ===")
dt_total = df['downtime_min'].sum()
planned  = df['planned_time_min'].sum()
print(f"  Total planned time   : {planned:,} min ({planned/60:,.0f} hours)")
print(f"  Total downtime       : {dt_total:,} min ({dt_total/planned*100:.1f}%)")
print(f"  Shifts with downtime : {(df['downtime_min']>0).sum():,} ({(df['downtime_min']>0).mean()*100:.1f}%)")

print("\n=== DEFECT SUMMARY ===")
total_units  = df['total_units'].sum()
total_rejects= df['reject_units'].sum()
print(f"  Total units produced : {total_units:,}")
print(f"  Total rejects        : {total_rejects:,} ({total_rejects/total_units*100:.2f}%)")

# ── 7. SAVE ──────────────────────────────────────────────────
df.to_csv("manufacturing_clean.csv", sep=",", decimal=".",
          index=False, encoding="utf-8-sig")
print("\n✅ Saved: manufacturing_clean.csv")

# Verify
df_check = pd.read_csv("manufacturing_clean.csv")
print(f"   Verification: {df_check.shape} · nulls={df_check.isnull().sum().sum()}")
print("⚠️  Copy via File Explorer ONLY — never open in Excel!")
