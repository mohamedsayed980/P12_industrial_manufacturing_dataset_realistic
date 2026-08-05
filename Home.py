"""
Repo_12_Industrial_Manufacturing — Home.py
Author : Mohamed · M3
"""
import pathlib
import streamlit as st

st.set_page_config(page_title="Industrial Manufacturing · M3",
                   page_icon="🏭", layout="wide")
LOGO = pathlib.Path(__file__).parent / "M3_logo.png"

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🏭 Industrial Manufacturing")
    st.markdown("M3 · ML Engine · P12")
    st.divider()
    st.markdown("**Navigate:**")
    st.markdown("📊 EDA Dashboard → 13 tabs")
    st.markdown("🤖 ML Models     → 5 tabs")

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1923;}
[data-testid="stSidebar"] *{color:#e0e8f0 !important;}
.main{background:#f4f7fb;}
.hero{background:linear-gradient(135deg,#1a237e,#00695c);
      padding:48px 40px;border-radius:14px;margin-bottom:28px;}
.hero h1{color:#ffffff !important;font-size:2.4rem;font-weight:800;margin:0 0 8px 0;}
.hero p{color:#b2dfdb !important;font-size:1.08rem;margin:0;}
.card{background:#ffffff;border-radius:10px;padding:22px 24px;
      box-shadow:0 2px 12px rgba(0,0,0,0.08);border-top:4px solid #00695c;}
.card h3{color:#00695c !important;margin:0 0 8px 0;font-size:1.05rem;}
.card p{color:#37474f !important;font-size:0.92rem;margin:0;line-height:1.6;}
.stat-card{background:#ffffff;border-radius:10px;padding:18px;text-align:center;
           box-shadow:0 2px 10px rgba(0,0,0,0.07);border-bottom:3px solid #00695c;}
.stat-num{font-size:1.9rem;font-weight:800;color:#00695c !important;}
.stat-lbl{font-size:0.82rem;color:#546e7a !important;margin-top:4px;}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🏭 Industrial Manufacturing Analytics</h1>
  <p>End-to-end OEE · Downtime · Defect · Production Analysis · M3 Portfolio · Project 12 of 12</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
for col, (num, lbl) in zip([c1,c2,c3,c4,c5,c6],[
    ("54,750","Shift Records"), ("365","Days (Full Year)"),
    ("50","Machines"), ("70.3%","Avg OEE"),
    ("5.57%","Reject Rate"), ("12.4%","Downtime Rate")]):
    col.markdown(f"""<div class="stat-card">
      <div class="stat-num">{num}</div>
      <div class="stat-lbl">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📌 About This Project")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""<div class="card"><h3>🎯 Objective</h3>
    <p>Full manufacturing analytics covering OEE optimization,
    downtime root cause analysis, defect classification, machine
    performance ranking, shift comparison, and energy efficiency.
    Predict and improve Overall Equipment Effectiveness.</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown("""<div class="card"><h3>📊 Dataset</h3>
    <p>54,750 shift records · 50 machines · 40 operators ·
    5 products · 3 shifts · Full year 2025.
    Covers: downtime reasons, defect types, OEE components
    (Availability × Performance × Quality).</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown("""<div class="card"><h3>🔑 Key Signals</h3>
    <p>No-downtime shifts have 80.3% OEE vs 70.2% with downtime.
    Only 7.5% of shifts reach world-class OEE ≥85%.
    30.9% of shifts classified as poor OEE &lt;65%.
    Reject rate: 5.57% = 1.82M wasted units.</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col4, col5 = st.columns(2)
with col4:
    st.markdown("### 📈 EDA Dashboard — 13 Tabs")
    for num, name, desc in [
        ("1","Data Overview","Shape, types, stats, dictionary"),
        ("2","OEE Dashboard ★","Full OEE breakdown + trend + category"),
        ("3","Downtime Analysis ★","Pareto + reason breakdown by machine"),
        ("4","Defect Analysis ★","Pareto + type by machine/product/shift"),
        ("5","Machine Performance ★","Ranking all 50 machines best→worst"),
        ("6","Shift Analysis ★","Shift_1 vs Shift_2 vs Shift_3 comparison"),
        ("7","Multicollinearity","VIF analysis"),
        ("8","Correlation","Heatmap + top OEE predictors"),
        ("9","Business KPIs ★","Cost of poor OEE · energy efficiency"),
        ("10","Category Deep-Dive ★","Machine × Shift × Product heatmaps"),
        ("11","Statistical Tests ★","T1-T4: shift OEE · downtime impact"),
        ("12","Feature Engineering","Engineered features + distributions"),
        ("13","Insights & Report","Findings + recommendations + download"),
    ]:
        st.markdown(f"**Tab {num} · {name}** — {desc}")

with col5:
    st.markdown("### 🤖 ML Models — 5 Tabs")
    for num, name, desc in [
        ("1","Model Training","6 Reg + 6 Clf · individual buttons"),
        ("2","Regression Results","R², MAE, RMSE · predict OEE"),
        ("3","Classification Results","F1, Recall, ROC-AUC · predict low OEE"),
        ("4","Feature Importance","Top OEE improvement drivers"),
        ("5","Predict","Interactive OEE predictor per shift"),
    ]:
        st.markdown(f"**Tab {num} · {name}** — {desc}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("**OEE = Availability × Performance × Quality**\n\n"
            "World class target: **≥ 85%**\n\n"
            "Current average: **70.3%** — 14.7 points below target\n\n"
            "Low OEE rate: **30.9%** → class_weight='balanced'")

st.markdown("---")
st.markdown("<p style='text-align:center;color:#90a4ae;font-size:0.85rem;'>"
            "Mohamed · M3 · ML Engine Portfolio · Project 12 of 12 · Industrial Manufacturing</p>",
            unsafe_allow_html=True)
