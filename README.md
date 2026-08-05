# 🏭 P12 — Industrial Manufacturing Analytics & OEE Prediction
**M3 · ML Engine Portfolio · Project 12 of 12**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Domain](https://img.shields.io/badge/Domain-Manufacturing-FF6B35)](https://github.com)

---

## 📌 Project Overview

End-to-end industrial manufacturing analytics on **54,750 production shift records** covering the full year 2025. This project goes beyond predictive maintenance (P11) to cover the complete manufacturing performance picture: OEE optimization, downtime root cause analysis, defect classification, machine ranking, shift comparison, and energy efficiency.

**Core Questions:**
- Why is average OEE only 70.3% when world-class is 85%?
- Which downtime causes drive the most availability loss?
- Which machines consistently underperform — and why?
- Can ML predict poor-OEE shifts before they start?
- What is the financial cost of the current OEE gap?

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Records | 54,750 shift records |
| Period | 2025-01-01 → 2025-12-31 (full year) |
| Machines | 50 |
| Operators | 40 |
| Products | 5 (P_A to P_E) |
| Shifts | 3 (Shift_1 / Shift_2 / Shift_3 — perfectly balanced) |
| Original Features | 18 |
| After Engineering | 38 |

### ⚠️ Informative Nulls — NOT Missing Data
```
downtime_reason = NaN (70%) → filled with 'No Downtime'
defect_type     = NaN (85%) → filled with 'No Defect'
```
These nulls mean the machine ran without downtime or defects — they carry real information.

---

## 🎯 Targets

| Type | Column | Description |
|------|--------|-------------|
| **Regression** | `OEE` | Overall Equipment Effectiveness (0.476–0.981) |
| **Classification** | `low_OEE` | 1 if OEE < 0.65 (poor performance) |

**OEE = Availability × Performance × Quality**

**Balance:** 30.9% low OEE / 69.1% normal → `class_weight='balanced'` needed

---

## 📊 Key Business Metrics

| Metric | Value | Insight |
|--------|-------|---------|
| Mean OEE | 70.3% | 14.7 pp below world-class 85% |
| World Class Shifts | 7.5% | Very few achieve excellence |
| Poor OEE Shifts | 30.9% | Nearly 1 in 3 shifts underperforms |
| Total Downtime | 12.4% of planned time | Primary availability loss |
| Reject Rate | 5.57% = 1.82M units | Major quality improvement opportunity |
| No-Downtime OEE | 80.3% vs 70.2% | Eliminating downtime = #1 OEE lever |

---

## ⚙️ Feature Engineering

| Feature | Formula | Insight |
|---------|---------|---------|
| `downtime_rate` | downtime_min / planned_time_min | 12.4% avg rate |
| `reject_rate` | reject_units / total_units | 5.57% avg rate |
| `energy_per_unit` | energy_kwh / good_units | 0.458 kWh/unit avg |
| `oee_gap` | 0.85 − OEE | Avg 14.7 pp gap from world class |
| `production_efficiency` | good_units / total_units | 94.4% avg |
| `no_downtime` | downtime_min == 0 | +10.1 pp OEE when no downtime |
| `has_breakdown` | downtime_reason == Breakdown | Tests breakdown OEE impact |
| `has_defect` | defect_type != No Defect | Tests defect quality impact |
| `low_OEE` | OEE < 0.65 | Classification target |
| `OEE_category` | 5 OEE bands | Critical/Poor/Fair/Good/World Class |
| `month`, `day_of_week` | Date features | Seasonal production patterns |

---

## 📊 EDA Dashboard — 13 Tabs

| Tab | Title | Highlight |
|-----|-------|-----------|
| 1 | Data Overview | Shape, types, stats, dictionary |
| 2 | OEE Dashboard ★ | Full breakdown + monthly trend + category distribution |
| 3 | Downtime Analysis ★ | Pareto chart + reason distribution + worst machines |
| 4 | Defect Analysis ★ | Pareto + defect type by product + worst machines |
| 5 | Machine Performance ★ | All 50 machines ranked best→worst by OEE |
| 6 | Shift Analysis ★ | Shift_1 vs Shift_2 vs Shift_3 full comparison |
| 7 | Multicollinearity | VIF analysis |
| 8 | Correlation | Heatmap + top OEE predictors |
| 9 | Business KPIs ★ | Downtime cost · defect cost · OEE improvement savings |
| 10 | Category Deep-Dive ★ | Machine × Shift × Product OEE heatmaps |
| 11 | Statistical Tests ★ | T1–T4: downtime · shifts · breakdown · defects |
| 12 | Feature Engineering | Engineered features + distributions |
| 13 | Insights & Report | Findings + report + Machine Rankings + Downtime KPIs download |

### Tab 13 Downloads (Option C — Value-Added Outputs)
- 📥 **Text Report** — executive summary with findings and recommendations
- 📥 **Machine Rankings** — all 50 machines ranked by OEE, downtime, reject rate
- 📥 **Downtime & Defect KPIs** — actionable maintenance and quality summary

---

## 🤖 ML Models — 5 Tabs

| Tab | Content |
|-----|---------|
| 1 | Training — 6 Reg + 6 Clf · individual buttons (never train all at once) |
| 2 | Regression Results — R², MAE, RMSE · predict OEE |
| 3 | Classification Results — F1, Recall, ROC-AUC · predict low OEE |
| 4 | Feature Importance — top OEE improvement drivers |
| 5 | Interactive OEE Predictor — shift risk scorer with factor analysis |

---

## 🔑 Key Findings

**1. Downtime is the #1 OEE Killer**
No-downtime shifts achieve 80.3% OEE vs 70.2% — a 10.1 percentage point difference. Eliminating downtime is the single biggest lever available.

**2. Only 7.5% of Shifts Reach World Class**
Despite 365 days of operation, only 4,027 of 54,750 shifts achieve ≥85% OEE. The improvement potential is enormous and largely untapped.

**3. Machine Variation is Significant**
OEE ranges from below 65% to above 85% across the 50 machines — the best machines are already demonstrating world-class performance. The gap is operational, not equipment.

**4. Maintenance Drives Most Downtime**
Maintenance and Breakdown together account for the majority of downtime hours. Shifting from reactive to preventive maintenance is the key operational change.

**5. 5.57% Reject Rate = 1.82M Wasted Units**
Scratch is the most common defect type — surface finishing and tooling processes need targeted quality improvement programs.

---

## 💡 Recommendations

| Priority | Action |
|----------|--------|
| 🔴 Critical | Schedule maintenance in lowest-demand shift to minimise availability loss |
| 🔴 Critical | Focus maintenance on bottom 10 machines — 2–3 pp OEE gain achievable |
| 🟡 High | Scratch reduction program — #1 defect type by volume |
| 🟡 High | Shift knowledge transfer — best shift practices to all shifts |
| 🟡 High | Energy benchmarking — best machines at 0.35 vs worst at 0.60+ kWh/unit |
| 🟢 Medium | Deploy ML model for real-time OEE risk flag before shift starts |

---

## 🗂 Project Structure

```
📁 Repo_12_Industrial_Manufacturing/
├── Home.py
├── M3_logo.png
├── requirements.txt
├── README.md
├── data/
│   └── manufacturing_clean.csv     ← from P12_clean_data.py (Jupyter)
└── pages/
    ├── EDA_dashboard.py             ← 13-tab analysis
    └── ML_Models.py                 ← 5-tab ML engine
```

---

## 🚀 How to Run

```bash
git clone https://github.com/YourUsername/Repo_12_Industrial_Manufacturing.git
cd Repo_12_Industrial_Manufacturing

pip install -r requirements.txt

# Step 1: Generate clean dataset in Jupyter
# Run P12_clean_data.py → saves manufacturing_clean.csv
# Copy to data/ folder via File Explorer (NEVER open in Excel)

# Step 2: Launch app
streamlit run Home.py
```

> ⚠️ **Critical:** Copy CSV via File Explorer only — never open in Excel before copying.

---

## 🛠 Tech Stack

`Python 3.11` · `Streamlit` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Plotly` · `Scikit-learn` · `SciPy` · `Statsmodels` · `Psutil`

---

**Mohamed · M3 · ML Engine Portfolio — 12 End-to-End Data Science Projects**
