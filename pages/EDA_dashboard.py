"""
Repo_12_Industrial_Manufacturing — EDA_dashboard.py  (13 Tabs)
Author : Mohamed · M3
Dataset: Industrial Manufacturing · 54,750 shift records · 2025
"""
import os, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st

warnings.filterwarnings("ignore")
S = st.session_state

st.set_page_config(page_title="EDA · Industrial Manufacturing · M3",
                   page_icon="🏭", layout="wide")

LOGO = pathlib.Path(__file__).parent.parent / "M3_logo.png"

_data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "manufacturing_clean.csv"
)

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🏭 EDA Dashboard")
    st.markdown("Industrial Manufacturing · 13 Tabs")
    st.divider()
    st.markdown("### 📂 Dataset")
    st.success("✅ manufacturing_clean.csv")
    st.caption("Loaded from data/ folder")

CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","light":"#e3f2fd","dark":"#1a237e",
       "purple":"#6a1b9a","amber":"#f57f17","grey":"#546e7a"}

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1923;}
[data-testid="stSidebar"] *{color:#e0e8f0 !important;}
.main{background:#f4f7fb;}
div[data-testid="metric-container"]{background:#e8f5e9;border-left:4px solid #00695c;border-radius:6px;padding:10px 14px;}
.sec-header{background:linear-gradient(90deg,#00695c,#1565c0);color:#ffffff !important;
  padding:10px 18px;border-radius:8px;font-size:1.1rem;font-weight:700;margin-bottom:16px;}
.insight-box{background:#e8f5e9;border-left:4px solid #2e7d32;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.insight-box p{color:#1b3a1f !important;margin:0;font-size:0.93rem;line-height:1.6;}
.warn-box{background:#fff3e0;border-left:4px solid #e65100;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.warn-box p{color:#4a2000 !important;margin:0;font-size:0.93rem;line-height:1.6;}
.info-box{background:#e3f2fd;border-left:4px solid #1565c0;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.info-box p{color:#0d2a4a !important;margin:0;font-size:0.93rem;line-height:1.6;}
</style>""", unsafe_allow_html=True)

def sec(t): st.markdown(f'<div class="sec-header">{t}</div>', unsafe_allow_html=True)
def insight(t): st.markdown(f'<div class="insight-box"><p>✅ {t}</p></div>', unsafe_allow_html=True)
def warn(t):    st.markdown(f'<div class="warn-box"><p>⚠️ {t}</p></div>', unsafe_allow_html=True)
def info(t):    st.markdown(f'<div class="info-box"><p>ℹ️ {t}</p></div>', unsafe_allow_html=True)

# ── LOAD ─────────────────────────────────────────────────────
if not os.path.exists(_data_path):
    st.error(f"❌ File not found: {_data_path}")
    st.info("Run P12_clean_data.py in Jupyter → copy manufacturing_clean.csv to data/ folder")
    st.stop()

df = pd.read_csv(_data_path, sep=",", decimal=".")
df.columns = df.columns.str.strip()
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

if df.empty:
    st.warning("⚠️ Dataset is empty."); st.stop()

S["df_work"] = df

TARGET  = "low_OEE"
REG_T   = "OEE"
WORLD_CLASS = 0.85
OEE_POOR    = 0.65

NUM_COLS = [c for c in ["OEE","availability","performance","quality",
                         "downtime_min","downtime_rate","reject_rate",
                         "energy_kwh","energy_per_unit","good_units",
                         "total_units","oee_gap","production_efficiency"]
            if c in df.columns]

DOWNTIME_REASONS = ["No Downtime","Maintenance","Breakdown","Material Shortage","Setup"]
DEFECT_TYPES     = ["No Defect","Scratch","Dimension Error","Crack","Surface Defect"]

# ── TABS ─────────────────────────────────────────────────────
tabs = st.tabs([
    "1 · Data Overview",
    "2 · OEE Dashboard ★",
    "3 · Downtime Analysis ★",
    "4 · Defect Analysis ★",
    "5 · Machine Performance ★",
    "6 · Shift Analysis ★",
    "7 · Multicollinearity",
    "8 · Correlation",
    "9 · Business KPIs ★",
    "10 · Category Deep-Dive ★",
    "11 · Statistical Tests ★",
    "12 · Feature Engineering",
    "13 · Insights & Report",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    sec("📋 Tab 1 — Data Overview")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Shift Records",  f"{len(df):,}")
    c2.metric("Machines",       f"{df['machine_id'].nunique()}")
    c3.metric("Operators",      f"{df['operator_id'].nunique()}")
    c4.metric("Products",       f"{df['product_id'].nunique()}")
    c5.metric("Mean OEE",       f"{df[REG_T].mean():.1%}")
    c6.metric("World Class %",  f"{(df[REG_T]>=WORLD_CLASS).mean():.1%}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📄 First 10 Rows")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        sec("📐 Column Info")
        info_df = pd.DataFrame({"Column":df.columns,
                                 "Dtype":df.dtypes.astype(str).values,
                                 "Nulls":df.isnull().sum().values})
        st.dataframe(info_df, use_container_width=True)

    st.markdown("---")
    sec("📊 Descriptive Statistics")
    st.dataframe(df[NUM_COLS].describe().round(4), use_container_width=True)

    st.markdown("---")
    sec("🗂 Data Dictionary")
    dd = pd.DataFrame({
        "Column":["date","shift","machine_id","product_id","operator_id",
                  "planned_time_min","downtime_min","downtime_reason",
                  "operating_time_min","total_units","good_units","reject_units",
                  "defect_type","energy_kwh","availability","performance","quality","OEE",
                  "downtime_rate","reject_rate","energy_per_unit","oee_gap",
                  "no_downtime","has_breakdown","has_defect","low_OEE","OEE_category"],
        "Type":  ["Date","Category","Category","Category","Category",
                  "Numeric","Numeric","Category","Numeric","Numeric","Numeric","Numeric",
                  "Category","Numeric","Numeric","Numeric","Numeric","Target(REG)",
                  "Engineered","Engineered","Engineered","Engineered",
                  "Engineered","Engineered","Engineered","Target(CLF)","Engineered"],
        "Description":[
            "Production date (2025-01-01 to 2025-12-31)",
            "Work shift: Shift_1 / Shift_2 / Shift_3",
            "Machine identifier (M01–M50)",
            "Product type (P_A to P_E)",
            "Operator identifier (OP001–OP040)",
            "Total planned shift time (always 480 min)",
            "Total downtime during shift (minutes)",
            "Downtime cause — NaN filled with 'No Downtime'",
            "Actual operating time = planned − downtime",
            "Total units attempted during shift",
            "Good (accepted) units produced",
            "Rejected/defective units",
            "Defect category — NaN filled with 'No Defect'",
            "Electricity consumed during shift (kWh)",
            "OEE component: operating_time / planned_time",
            "OEE component: actual output rate / max rate",
            "OEE component: good_units / total_units",
            "OEE = Availability × Performance × Quality — REG TARGET",
            "downtime_min / planned_time_min",
            "reject_units / total_units",
            "energy_kwh / good_units (kWh per good unit)",
            "0.85 − OEE (gap from world class)",
            "1 if downtime_min = 0",
            "1 if downtime_reason = Breakdown",
            "1 if defect_type ≠ No Defect",
            "1 if OEE < 0.65 (poor performance) — CLF TARGET",
            "Critical/Poor/Fair/Good/World Class",
        ]
    })
    st.dataframe(dd, use_container_width=True)
    warn("30.9% of shifts have poor OEE < 65% → class_weight='balanced' on all classifiers")

# ══════════════════════════════════════════════════════════════
# TAB 2 — OEE DASHBOARD ★
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    sec("📊 Tab 2 — OEE Dashboard ★")
    info("OEE = Availability × Performance × Quality. World class target = 85%.")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Mean OEE",          f"{df['OEE'].mean():.1%}")
    c2.metric("Mean Availability",  f"{df['availability'].mean():.1%}")
    c3.metric("Mean Performance",   f"{df['performance'].mean():.1%}")
    c4.metric("Mean Quality",       f"{df['quality'].mean():.1%}")
    c5.metric("World Class Shifts", f"{(df['OEE']>=WORLD_CLASS).mean():.1%}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 OEE Category Distribution")
        if "OEE_category" in df.columns:
            oee_cat = df["OEE_category"].value_counts().sort_index()
            colors_cat = [CLR["danger"],CLR["warning"],CLR["amber"],
                          CLR["success"],CLR["teal"]]
            fig = px.bar(x=oee_cat.index.astype(str), y=oee_cat.values,
                         color=oee_cat.index.astype(str),
                         color_discrete_sequence=colors_cat,
                         title="OEE Category Distribution",
                         text=oee_cat.values)
            fig.update_traces(textposition="outside")
            fig.update_layout(height=380, showlegend=False,
                              xaxis_title="OEE Category", yaxis_title="Shift Count")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 OEE Distribution")
        fig2, ax = plt.subplots(figsize=(7,4))
        ax.hist(df["OEE"], bins=60, color=CLR["teal"], edgecolor="white", alpha=0.85)
        ax.axvline(WORLD_CLASS, color=CLR["success"], lw=2.5, ls="--",
                   label=f"World Class ({WORLD_CLASS:.0%})")
        ax.axvline(OEE_POOR,    color=CLR["danger"],  lw=2.5, ls="--",
                   label=f"Poor OEE ({OEE_POOR:.0%})")
        ax.axvline(df["OEE"].mean(), color=CLR["primary"], lw=2,
                   label=f"Mean ({df['OEE'].mean():.1%})")
        ax.set_xlabel("OEE"); ax.set_ylabel("Shift Count")
        ax.set_title("OEE Distribution", fontweight="bold"); ax.legend()
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    sec("📊 OEE Monthly Trend (2025)")
    monthly = df.groupby("month")[["OEE","availability","performance","quality"]].mean()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    fig3 = go.Figure()
    for col, color in [("OEE",CLR["teal"]),("availability",CLR["primary"]),
                       ("performance",CLR["amber"]),("quality",CLR["success"])]:
        fig3.add_trace(go.Scatter(
            x=[month_names[m] for m in monthly.index],
            y=monthly[col], mode="lines+markers",
            name=col.capitalize(), line=dict(color=color, width=2.5)))
    fig3.add_hline(y=WORLD_CLASS, line_dash="dash", line_color=CLR["success"],
                   annotation_text="World Class 85%")
    fig3.update_layout(height=380, title="Monthly OEE Components Trend",
                       yaxis_tickformat=".0%")
    st.plotly_chart(fig3, use_container_width=True)

    insight(f"Mean OEE {df['OEE'].mean():.1%} — {(WORLD_CLASS-df['OEE'].mean())*100:.1f} percentage points below world-class target.")
    insight("No-downtime shifts achieve 80.3% OEE vs 70.2% with downtime — availability is the primary lever.")
    warn("Only 7.5% of shifts reach world-class OEE ≥85% — significant improvement potential exists.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — DOWNTIME ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    sec("⏱ Tab 3 — Downtime Analysis ★")
    info("12.4% of planned time is lost to downtime. Eliminating downtime is the #1 OEE improvement lever.")

    total_dt   = df["downtime_min"].sum()
    planned    = df["planned_time_min"].sum()
    shifts_dt  = (df["downtime_min"] > 0).sum()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Downtime",      f"{total_dt/60:,.0f} hrs")
    c2.metric("Downtime Rate",        f"{total_dt/planned:.1%}")
    c3.metric("Shifts with Downtime", f"{shifts_dt:,}")
    c4.metric("Avg Downtime/Shift",   f"{df['downtime_min'].mean():.0f} min")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Downtime Pareto — by Reason")
        dt_reason = df.groupby("downtime_reason")["downtime_min"].sum()\
                      .sort_values(ascending=False)
        dt_reason = dt_reason[dt_reason.index != "No Downtime"]
        dt_pct    = (dt_reason / dt_reason.sum() * 100).round(1)
        fig, ax   = plt.subplots(figsize=(7,4))
        bars = ax.bar(dt_reason.index, dt_reason.values/60,
                      color=[CLR["danger"],CLR["warning"],CLR["amber"],CLR["primary"]],
                      edgecolor="white")
        ax2  = ax.twinx()
        cumulative = dt_pct.cumsum()
        ax2.plot(dt_reason.index, cumulative.values, "o--",
                 color=CLR["dark"], lw=2, ms=8, label="Cumulative %")
        ax2.axhline(80, color=CLR["grey"], ls=":", lw=1.5, label="80% line")
        ax2.set_ylabel("Cumulative %"); ax2.set_ylim(0,105)
        ax.set_ylabel("Downtime (hours)"); ax.set_xlabel("Downtime Reason")
        ax.set_title("Downtime Pareto Chart", fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        for bar,val in zip(bars, dt_reason.values):
            ax.text(bar.get_x()+bar.get_width()/2, val/60+50,
                    f"{val/60:,.0f}h", ha="center", fontsize=8)
        ax2.legend(loc="center right")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Downtime Reason Distribution")
        dt_count = df["downtime_reason"].value_counts()
        fig2 = px.pie(values=dt_count.values, names=dt_count.index,
                      color_discrete_sequence=[CLR["success"],CLR["danger"],
                                               CLR["warning"],CLR["amber"],CLR["primary"]],
                      title="Shift Count by Downtime Reason", hole=0.4)
        fig2.update_traces(textinfo="percent+label")
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    sec("📊 Average Downtime by Machine — Top 20 Worst")
    machine_dt = df.groupby("machine_id")["downtime_min"].mean()\
                   .sort_values(ascending=False).head(20)
    fig3 = px.bar(x=machine_dt.index, y=machine_dt.values,
                  color=machine_dt.values,
                  color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                  title="Avg Downtime per Shift — Top 20 Worst Machines",
                  text=machine_dt.values.round(0))
    fig3.update_traces(textposition="outside")
    fig3.update_layout(height=380, xaxis_title="Machine", yaxis_title="Avg Downtime (min)")
    st.plotly_chart(fig3, use_container_width=True)

    insight("Maintenance and Breakdown together account for the majority of productive time loss.")
    insight("70% of shifts have NO downtime — when downtime occurs, it hits hard on OEE.")
    warn("Focus maintenance resources on the top 5 worst machines — Pareto principle applies here.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — DEFECT ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    sec("🔍 Tab 4 — Defect Analysis ★")
    info("5.57% reject rate = 1.82 million wasted units over 2025. Each defect type has a different root cause.")

    total_units  = df["total_units"].sum()
    total_reject = df["reject_units"].sum()
    defect_shifts= (df["defect_type"] != "No Defect").sum()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Units",    f"{total_units/1e6:.2f}M")
    c2.metric("Total Rejects",  f"{total_reject/1e6:.2f}M")
    c3.metric("Reject Rate",    f"{total_reject/total_units:.2%}")
    c4.metric("Defect Shifts",  f"{defect_shifts:,} ({defect_shifts/len(df):.1%})")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Defect Type Pareto")
        df_defects = df[df["defect_type"] != "No Defect"]
        defect_cnt = df_defects["defect_type"].value_counts()
        defect_pct = (defect_cnt / defect_cnt.sum() * 100).round(1)
        fig, ax    = plt.subplots(figsize=(7,4))
        bars = ax.bar(defect_cnt.index, defect_cnt.values,
                      color=[CLR["danger"],CLR["warning"],CLR["amber"],CLR["primary"]],
                      edgecolor="white")
        ax2  = ax.twinx()
        ax2.plot(defect_cnt.index, defect_pct.cumsum().values, "o--",
                 color=CLR["dark"], lw=2, ms=8)
        ax2.axhline(80, color=CLR["grey"], ls=":", lw=1.5)
        ax2.set_ylabel("Cumulative %"); ax2.set_ylim(0,105)
        ax.set_ylabel("Defect Count"); ax.set_title("Defect Type Pareto", fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Reject Rate by Product")
        prod_reject = df.groupby("product_id").apply(
            lambda x: x["reject_units"].sum() / x["total_units"].sum() * 100
        ).round(2).sort_values(ascending=False)
        fig2 = px.bar(x=prod_reject.index, y=prod_reject.values,
                      color=prod_reject.values,
                      color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Reject Rate % by Product",
                      text=prod_reject.values.round(1))
        fig2.update_traces(textposition="outside",
                           texttemplate="%{text}%")
        fig2.update_layout(height=380, xaxis_title="Product",
                           yaxis_title="Reject Rate %")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    sec("📊 Defect Rate by Machine — Top 20 Worst")
    machine_reject = df.groupby("machine_id").apply(
        lambda x: x["reject_units"].sum() / x["total_units"].sum() * 100
    ).sort_values(ascending=False).head(20)
    fig3 = px.bar(x=machine_reject.index, y=machine_reject.values,
                  color=machine_reject.values,
                  color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                  title="Defect Rate % — Top 20 Worst Machines",
                  text=machine_reject.values.round(1))
    fig3.update_traces(textposition="outside", texttemplate="%{text}%")
    fig3.update_layout(height=380)
    st.plotly_chart(fig3, use_container_width=True)

    insight("Scratch is the most common defect — surface finishing process needs review.")
    warn(f"Top defect machines have 2-3× the average reject rate — targeted quality intervention needed.")

# ══════════════════════════════════════════════════════════════
# TAB 5 — MACHINE PERFORMANCE ★
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    sec("🤖 Tab 5 — Machine Performance ★")
    info("Ranking all 50 machines by OEE — identifies which machines need priority attention.")

    machine_oee = df.groupby("machine_id")["OEE"].mean()\
                    .sort_values(ascending=False).round(4)
    machine_avail = df.groupby("machine_id")["availability"].mean().round(4)
    machine_qual  = df.groupby("machine_id")["quality"].mean().round(4)

    col1, col2 = st.columns(2)
    with col1:
        sec("🏆 Top 10 Best Machines")
        top10 = machine_oee.head(10).reset_index()
        top10.columns = ["Machine","Avg OEE"]
        top10["OEE%"] = (top10["Avg OEE"]*100).round(1).astype(str)+"%"
        st.dataframe(top10.style.background_gradient(subset=["Avg OEE"],cmap="Greens"),
                     use_container_width=True)
    with col2:
        sec("⚠️ Bottom 10 Worst Machines")
        bot10 = machine_oee.tail(10).reset_index()
        bot10.columns = ["Machine","Avg OEE"]
        bot10["OEE%"] = (bot10["Avg OEE"]*100).round(1).astype(str)+"%"
        st.dataframe(bot10.style.background_gradient(subset=["Avg OEE"],cmap="Reds_r"),
                     use_container_width=True)

    st.markdown("---")
    sec("📊 All 50 Machines — OEE Ranking")
    fig = px.bar(x=machine_oee.index, y=machine_oee.values,
                 color=machine_oee.values,
                 color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                 title="OEE Ranking — All 50 Machines",
                 labels={"x":"Machine","y":"Avg OEE"})
    fig.add_hline(y=WORLD_CLASS, line_dash="dash", line_color=CLR["success"],
                  annotation_text="World Class 85%")
    fig.add_hline(y=OEE_POOR,   line_dash="dash", line_color=CLR["danger"],
                  annotation_text="Poor OEE 65%")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    oee_range = machine_oee.max() - machine_oee.min()
    insight(f"OEE range across machines: {machine_oee.min():.1%} to {machine_oee.max():.1%} — {oee_range:.1%} spread shows significant variation.")
    insight("Best machines can teach us what practices achieve world-class OEE — benchmark and replicate.")
    warn("Bottom 10 machines drag the overall average down — targeted maintenance/upgrade needed.")

# ══════════════════════════════════════════════════════════════
# TAB 6 — SHIFT ANALYSIS ★
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    sec("🔄 Tab 6 — Shift Analysis ★")
    info("Shift_1 vs Shift_2 vs Shift_3 — which shift performs best and why?")

    shift_stats = df.groupby("shift")[
        ["OEE","availability","performance","quality",
         "downtime_min","reject_rate","energy_kwh"]
    ].mean().round(4)

    st.dataframe(shift_stats.style.background_gradient(subset=["OEE"],cmap="RdYlGn"),
                 use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 OEE by Shift")
        shift_oee = df.groupby("shift")["OEE"].mean()
        fig = px.bar(x=shift_oee.index, y=shift_oee.values,
                     color=shift_oee.values,
                     color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                     title="Average OEE by Shift",
                     text=shift_oee.values.round(3))
        fig.add_hline(y=df["OEE"].mean(), line_dash="dash",
                      line_color=CLR["primary"],
                      annotation_text=f"Overall avg {df['OEE'].mean():.1%}")
        fig.update_traces(textposition="outside",
                          texttemplate="%{text:.1%}")
        fig.update_layout(height=370, showlegend=False,
                          yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 OEE Box Plot by Shift")
        fig2, ax = plt.subplots(figsize=(7,4))
        shifts = sorted(df["shift"].unique())
        data_bp = [df[df["shift"]==s]["OEE"].values for s in shifts]
        bp = ax.boxplot(data_bp, patch_artist=True, labels=shifts)
        colors_bp = [CLR["primary"],CLR["teal"],CLR["amber"]]
        for patch, color in zip(bp["boxes"], colors_bp):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color(CLR["danger"]); median.set_linewidth(2)
        ax.axhline(WORLD_CLASS, color=CLR["success"], ls="--", lw=1.5,
                   label="World class 85%")
        ax.axhline(OEE_POOR, color=CLR["danger"], ls="--", lw=1.5,
                   label="Poor OEE 65%")
        ax.set_ylabel("OEE"); ax.set_title("OEE Distribution by Shift", fontweight="bold")
        ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        sec("📊 Downtime by Shift")
        shift_dt = df.groupby("shift")["downtime_min"].mean()
        fig3 = px.bar(x=shift_dt.index, y=shift_dt.values,
                      color=shift_dt.values,
                      color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Avg Downtime per Shift (min)",
                      text=shift_dt.values.round(0))
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        sec("📊 Reject Rate by Shift")
        shift_rr = df.groupby("shift")["reject_rate"].mean() * 100
        fig4 = px.bar(x=shift_rr.index, y=shift_rr.values,
                      color=shift_rr.values,
                      color_continuous_scale=["#2e7d32","#e65100","#c62828"],
                      title="Avg Reject Rate % by Shift",
                      text=shift_rr.values.round(2))
        fig4.update_traces(textposition="outside",
                           texttemplate="%{text:.2f}%")
        fig4.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    best_shift  = shift_stats["OEE"].idxmax()
    worst_shift = shift_stats["OEE"].idxmin()
    insight(f"{best_shift} has the highest average OEE — study its practices and replicate.")
    insight(f"{worst_shift} has the lowest OEE — investigate operator scheduling and maintenance timing.")

# ══════════════════════════════════════════════════════════════
# TAB 7 — MULTICOLLINEARITY
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    sec("🔁 Tab 7 — Multicollinearity / VIF")
    info("OEE = Availability × Performance × Quality — high VIF expected between these components.")

    vif_cols = [c for c in ["availability","performance","quality",
                             "downtime_rate","reject_rate","energy_per_unit",
                             "no_downtime","has_breakdown","has_defect",
                             "oee_gap","production_efficiency"]
                if c in df.columns]
    vif_data = df[vif_cols].dropna()
    try:
        vif_df = pd.DataFrame({
            "Feature": vif_cols,
            "VIF": [round(variance_inflation_factor(vif_data.values,i),2)
                    for i in range(len(vif_cols))]
        }).sort_values("VIF", ascending=False)
        vif_df["Risk"] = vif_df["VIF"].apply(
            lambda v: "🔴 High" if v>10 else "🟡 Medium" if v>5 else "🟢 Low")

        col1, col2 = st.columns([1,1.5])
        with col1: st.dataframe(vif_df, use_container_width=True)
        with col2:
            fig, ax = plt.subplots(figsize=(7,5))
            colors_vif = [CLR["danger"] if v>10 else CLR["warning"] if v>5
                          else CLR["success"] for v in vif_df["VIF"]]
            ax.barh(vif_df["Feature"], vif_df["VIF"], color=colors_vif)
            ax.axvline(10, color=CLR["danger"],  lw=2, ls="--", label="VIF=10")
            ax.axvline(5,  color=CLR["warning"], lw=1.5, ls=":",  label="VIF=5")
            ax.set_xlabel("VIF"); ax.set_title("Multicollinearity Check")
            ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()
    except Exception as e:
        warn(f"VIF error: {e}")

    warn("quality ≈ production_efficiency — mathematically related. Use only one for linear models.")
    warn("OEE is derived from availability × performance × quality — expect high VIF.")
    insight("Tree models (RF, GB) handle multicollinearity automatically — preferred for this dataset.")

# ══════════════════════════════════════════════════════════════
# TAB 8 — CORRELATION
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    sec("🔥 Tab 8 — Correlation Analysis")

    num_corr = [c for c in df.select_dtypes(include=np.number).columns
                if c not in ["low_OEE","month","day_of_week","week",
                              "is_weekend","planned_time_min"]]
    corr = df[num_corr].corr()

    fig, ax = plt.subplots(figsize=(14,10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size":7})
    ax.set_title("Correlation Matrix — Manufacturing Features",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    sec("🎯 Top Correlations with OEE")
    tgt = corr["OEE"].drop("OEE").sort_values(key=abs, ascending=False).head(12)
    fig2, ax2 = plt.subplots(figsize=(10,5))
    colors_bar = [CLR["success"] if v>0 else CLR["danger"] for v in tgt.values]
    ax2.barh(tgt.index, tgt.values, color=colors_bar)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("Pearson r with OEE")
    ax2.set_title("Feature Correlations with OEE", fontsize=12, fontweight="bold")
    for i,(idx,val) in enumerate(tgt.items()):
        ax2.text(val+0.005 if val>=0 else val-0.005, i,
                 f"{val:.3f}", va="center",
                 ha="left" if val>=0 else "right", fontsize=9)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    insight("Availability is the strongest OEE predictor — reducing downtime has the biggest impact.")
    warn("OEE components (availability, performance, quality) show high correlation with OEE by definition.")

# ══════════════════════════════════════════════════════════════
# TAB 9 — BUSINESS KPIs ★
# ══════════════════════════════════════════════════════════════
with tabs[8]:
    sec("💼 Tab 9 — Business KPIs ★")
    info("Quantifying the financial impact of poor OEE, downtime, and defects.")

    COST_PER_HOUR_DOWNTIME = 500
    COST_PER_REJECT_UNIT   = 15
    ENERGY_COST_PER_KWH    = 0.12

    total_downtime_hrs = df["downtime_min"].sum() / 60
    total_reject       = df["reject_units"].sum()
    total_energy       = df["energy_kwh"].sum()

    downtime_cost = total_downtime_hrs * COST_PER_HOUR_DOWNTIME
    reject_cost   = total_reject * COST_PER_REJECT_UNIT
    energy_cost   = total_energy * ENERGY_COST_PER_KWH
    total_cost    = downtime_cost + reject_cost

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Downtime Cost",  f"${downtime_cost/1e6:.2f}M")
    c2.metric("Reject Cost",    f"${reject_cost/1e6:.2f}M")
    c3.metric("Total Loss",     f"${total_cost/1e6:.2f}M")
    c4.metric("Energy Cost",    f"${energy_cost/1e6:.2f}M")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Cost Breakdown")
        cost_df = pd.DataFrame({
            "Category":  ["Downtime Loss","Reject/Scrap","Energy Cost"],
            "Cost ($M)": [downtime_cost/1e6, reject_cost/1e6, energy_cost/1e6]
        })
        fig = px.bar(cost_df, x="Category", y="Cost ($M)",
                     color="Category",
                     color_discrete_map={"Downtime Loss":CLR["danger"],
                                         "Reject/Scrap":CLR["warning"],
                                         "Energy Cost":CLR["primary"]},
                     title="Annual Cost Breakdown",
                     text=cost_df["Cost ($M)"].apply(lambda x: f"${x:.2f}M"))
        fig.update_traces(textposition="outside")
        fig.update_layout(height=370, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sec("📊 OEE Improvement Savings Potential")
        current_oee = df["OEE"].mean()
        savings_scenarios = []
        for target in [0.75, 0.80, 0.85]:
            oee_gain     = target - current_oee
            dt_reduction = df["downtime_min"].sum() * (oee_gain / (1-current_oee)) / 60
            saving       = dt_reduction * COST_PER_HOUR_DOWNTIME
            savings_scenarios.append({"Target OEE": f"{target:.0%}",
                                       "OEE Gain": f"+{oee_gain:.1%}",
                                       "Saving ($M)": round(saving/1e6,2)})
        sav_df = pd.DataFrame(savings_scenarios)
        st.dataframe(sav_df, use_container_width=True)

        fig2 = px.bar(sav_df, x="Target OEE", y="Saving ($M)",
                      color="Saving ($M)",
                      color_continuous_scale=["#e65100","#2e7d32"],
                      title="Savings by OEE Target",
                      text=sav_df["Saving ($M)"].apply(lambda x: f"${x:.2f}M"))
        fig2.update_traces(textposition="outside")
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    insight(f"Reaching 85% OEE would save an estimated ${(0.85-current_oee)/0.297*downtime_cost/1e6:.1f}M annually in downtime costs.")
    insight(f"Energy per good unit = {df['energy_per_unit'].mean():.3f} kWh — benchmark against best machines to find efficiency gains.")

# ══════════════════════════════════════════════════════════════
# TAB 10 — CATEGORY DEEP-DIVE ★
# ══════════════════════════════════════════════════════════════
with tabs[9]:
    sec("🔎 Tab 10 — Category Deep-Dive ★")
    info("Cross-tabulation: which combination of machine × shift × product drives the lowest OEE?")

    sec("📊 Machine × Shift — OEE Heatmap (Top 15 Machines)")
    top15_machines = df.groupby("machine_id")["OEE"].mean()\
                       .sort_values().head(15).index
    heat = df[df["machine_id"].isin(top15_machines)]\
             .groupby(["machine_id","shift"])["OEE"].mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(heat.round(1), annot=True, fmt=".1f", cmap="RdYlGn",
                ax=ax, linewidths=0.5, annot_kws={"size":9})
    ax.set_title("OEE % — Bottom 15 Machines × Shift", fontsize=12, fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Product × Shift — OEE")
        heat2 = df.groupby(["product_id","shift"])["OEE"].mean().unstack() * 100
        fig2, ax2 = plt.subplots(figsize=(7,4))
        sns.heatmap(heat2.round(1), annot=True, fmt=".1f", cmap="RdYlGn",
                    ax=ax2, linewidths=0.5, annot_kws={"size":10})
        ax2.set_title("OEE % — Product × Shift", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    with col2:
        sec("📊 Downtime Reason × Shift")
        heat3 = df.groupby(["downtime_reason","shift"])["downtime_min"].mean().unstack()
        heat3 = heat3.drop("No Downtime", errors="ignore")
        fig3, ax3 = plt.subplots(figsize=(7,4))
        sns.heatmap(heat3.round(0), annot=True, fmt=".0f", cmap="YlOrRd",
                    ax=ax3, linewidths=0.5, annot_kws={"size":10})
        ax3.set_title("Avg Downtime (min) — Reason × Shift", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    insight("Certain machine × shift combinations consistently underperform — scheduling and operator assignment matter.")
    warn("Products with highest reject rates in specific shifts suggest operator skill or tooling issues.")

# ══════════════════════════════════════════════════════════════
# TAB 11 — STATISTICAL TESTS ★
# ══════════════════════════════════════════════════════════════
with tabs[10]:
    sec("🧪 Tab 11 — Statistical Tests ★")
    info("4 tests validating the most important manufacturing performance questions.")

    def run_ab(gA, gB, lA, lB, metric):
        t_stat, p_val = stats.ttest_ind(gA.dropna(), gB.dropna(), equal_var=False)
        pooled   = np.sqrt((gA.std()**2 + gB.std()**2) / 2)
        cohens_d = (gA.mean() - gB.mean()) / (pooled + 1e-10)
        res = pd.DataFrame({
            "Metric":["Test","Group A","Group B","A Mean","B Mean",
                      "t-stat","p-value","Significant","Cohen's d","Effect","Decision"],
            "Result":["Welch T-Test",
                      f"{lA} (n={len(gA):,})", f"{lB} (n={len(gB):,})",
                      f"{gA.mean():.4f}", f"{gB.mean():.4f}",
                      f"{t_stat:.4f}", f"{p_val:.6f}",
                      "✅ YES" if p_val<0.05 else "❌ NO",
                      f"{cohens_d:.4f}",
                      "Large" if abs(cohens_d)>0.8 else "Medium" if abs(cohens_d)>0.5 else "Small",
                      "✅ REJECT H₀" if p_val<0.05 else "❌ FAIL to reject H₀"]
        })
        return res, p_val, cohens_d

    # T1: No Downtime vs With Downtime
    sec("T1 — Does eliminating downtime significantly improve OEE?")
    r1, p1, d1 = run_ab(df[df["no_downtime"]==1]["OEE"],
                         df[df["no_downtime"]==0]["OEE"],
                         "No Downtime","With Downtime","OEE")
    col1, col2 = st.columns([1.2,1])
    with col1: st.dataframe(r1, use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.hist(df[df["no_downtime"]==0]["OEE"], bins=40, alpha=0.6,
                color=CLR["danger"],  density=True, label="With Downtime")
        ax.hist(df[df["no_downtime"]==1]["OEE"], bins=40, alpha=0.7,
                color=CLR["success"], density=True, label="No Downtime")
        ax.set_title("T1: OEE Distribution"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()
    if p1 < 0.05:
        insight(f"T1: No-downtime shifts have significantly higher OEE (d={d1:.3f}) — availability is the #1 lever.")

    # T2: Shift comparison (ANOVA)
    st.markdown("---")
    sec("T2 — Do the 3 shifts have significantly different OEE? (ANOVA)")
    from scipy.stats import f_oneway
    shifts_data = [df[df["shift"]==s]["OEE"] for s in sorted(df["shift"].unique())]
    f_stat, p_anova = f_oneway(*shifts_data)
    st.markdown(f"**ANOVA F-statistic:** {f_stat:.4f} · **p-value:** {p_anova:.6f}")
    st.markdown(f"**Significant:** {'✅ YES — shifts differ' if p_anova < 0.05 else '❌ NO'}")
    if p_anova < 0.05:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey = pairwise_tukeyhsd(df["OEE"], df["shift"], alpha=0.05)
        st.text(str(tukey.summary()))
        insight("Shift OEE differences are statistically significant — investigate scheduling, operator assignment, and handover quality.")

    # T3: Breakdown vs No Breakdown
    st.markdown("---")
    sec("T3 — Does a Breakdown event significantly impact OEE?")
    r3, p3, d3 = run_ab(df[df["has_breakdown"]==1]["OEE"],
                         df[df["has_breakdown"]==0]["OEE"],
                         "Has Breakdown","No Breakdown","OEE")
    st.dataframe(r3, use_container_width=True)
    if p3 < 0.05:
        insight(f"T3: Breakdown shifts have significantly lower OEE (d={d3:.3f}).")
    else:
        info("T3: Breakdown impact on OEE is not statistically significant — breakdowns are already short-duration events.")

    # T4: Defect vs No Defect
    st.markdown("---")
    sec("T4 — Does having defects significantly reduce OEE quality component?")
    r4, p4, d4 = run_ab(df[df["has_defect"]==1]["quality"],
                         df[df["has_defect"]==0]["quality"],
                         "Has Defect","No Defect","Quality")
    st.dataframe(r4, use_container_width=True)
    if p4 < 0.05:
        insight(f"T4: Defect shifts have significantly lower quality component (d={d4:.3f}).")

# ══════════════════════════════════════════════════════════════
# TAB 12 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
with tabs[11]:
    sec("⚙️ Tab 12 — Feature Engineering")

    fe = pd.DataFrame({
        "Feature":      ["downtime_rate","reject_rate","energy_per_unit",
                         "oee_gap","production_efficiency",
                         "no_downtime","has_breakdown","has_defect",
                         "low_OEE","OEE_category","month","day_of_week","is_weekend"],
        "Formula":      ["downtime_min / planned_time_min",
                         "reject_units / total_units",
                         "energy_kwh / good_units",
                         "0.85 − OEE",
                         "good_units / total_units",
                         "1 if downtime_min = 0",
                         "1 if downtime_reason = Breakdown",
                         "1 if defect_type ≠ No Defect",
                         "1 if OEE < 0.65",
                         "pd.cut into 5 OEE bands",
                         "date.dt.month","date.dt.dayofweek","dayofweek >= 5"],
        "Insight":      ["12.4% avg downtime rate — primary availability loss",
                         "5.57% avg reject rate — 1.82M units wasted",
                         "0.458 kWh/unit avg — benchmark against best machines",
                         "Avg gap = 14.7% below world class",
                         "94.4% avg — good output when running",
                         "No-downtime shifts: OEE 80.3% vs 70.2%",
                         "Tests if breakdowns uniquely impact OEE",
                         "Tests if defect presence affects OEE quality",
                         "30.9% poor OEE — CLF target",
                         "5-level categorical for segmentation",
                         "Monthly seasonality in production",
                         "Day-of-week production patterns",
                         "Weekend vs weekday performance"],
    })
    st.dataframe(fe, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        sec("📊 Downtime Rate Distribution")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.hist(df["downtime_rate"], bins=50, color=CLR["danger"],
                edgecolor="white", alpha=0.85)
        ax.axvline(df["downtime_rate"].mean(), color="black", lw=2,
                   label=f"Mean={df['downtime_rate'].mean():.1%}")
        ax.set_xlabel("Downtime Rate"); ax.set_title("Downtime Rate Distribution")
        ax.legend(); plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        sec("📊 Energy per Unit Distribution")
        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.hist(df["energy_per_unit"].clip(0,2), bins=50, color=CLR["teal"],
                 edgecolor="white", alpha=0.85)
        ax2.axvline(df["energy_per_unit"].mean(), color="black", lw=2,
                    label=f"Mean={df['energy_per_unit'].mean():.3f}")
        ax2.set_xlabel("kWh per Good Unit"); ax2.set_title("Energy Efficiency Distribution")
        ax2.legend(); plt.tight_layout(); st.pyplot(fig2); plt.close()

    insight("no_downtime flag is the most impactful binary feature — 10.1 percentage point OEE difference.")
    warn("Informative nulls (70% downtime_reason, 85% defect_type) were filled with 'No Downtime'/'No Defect' — not removed.")

# ══════════════════════════════════════════════════════════════
# TAB 13 — INSIGHTS & REPORT
# ══════════════════════════════════════════════════════════════
with tabs[12]:
    sec("💡 Tab 13 — Insights & Recommendations")

    current_oee = df["OEE"].mean()
    st.markdown(f"### 🏭 Industrial Manufacturing — Final Report")
    st.markdown(f"**54,750 shift records · {current_oee:.1%} avg OEE · 50 machines · 40 operators · M3**")
    st.markdown("---")

    sec("1️⃣ OEE Performance Gap")
    insight(f"Current OEE: {current_oee:.1%} — {(WORLD_CLASS-current_oee)*100:.1f} percentage points below world-class target of 85%.")
    insight(f"Only 7.5% of shifts achieve world-class OEE — significant and achievable improvement potential.")
    warn(f"30.9% of shifts fall in poor OEE zone (<65%) — nearly 1 in 3 shifts is underperforming.")

    sec("2️⃣ Top OEE Killers")
    insight("Downtime is #1 OEE killer — no-downtime shifts achieve 80.3% vs 70.2% (10.1 pp difference).")
    insight("Maintenance and Breakdown together are the dominant downtime causes — scheduled PM is key.")
    insight("5.57% reject rate = 1.82 million wasted units — quality improvement opportunity.")

    sec("3️⃣ Recommendations")
    recs = [
        ("📅 Implement Preventive Maintenance",
         "Schedule maintenance in Shift_3 (typically lowest production demand) to minimize availability loss."),
        ("🔧 Focus on Top 10 Worst Machines",
         "Targeted maintenance/upgrade of bottom 10 machines can raise overall OEE by 2-3 percentage points."),
        ("🎯 Scratch Reduction Program",
         "Scratch is the #1 defect type — investigate tooling condition and surface preparation processes."),
        ("📊 Shift Knowledge Transfer",
         "Best-performing shift achieves higher OEE — document their practices and train other shifts."),
        ("⚡ Energy Benchmarking",
         "Best machines produce at 0.35 kWh/unit vs worst at 0.60+ kWh/unit — replicate efficient practices."),
        ("🤖 ML Deployment",
         "Deploy OEE prediction model to flag shifts likely to underperform before they start."),
    ]
    for title, text in recs:
        st.markdown(f'<div class="warn-box"><p><b>{title}:</b> {text}</p></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    report_txt = f"""INDUSTRIAL MANUFACTURING — FINAL REPORT
M3 · 54,750 Shift Records · 50 Machines · 2025

OEE PERFORMANCE:
  Current Mean OEE    : {current_oee:.1%}
  World Class Target  : 85%
  Gap                 : {(WORLD_CLASS-current_oee)*100:.1f} percentage points
  World Class Shifts  : 7.5% only
  Poor OEE Shifts     : 30.9% (< 65%)

DOWNTIME:
  Total downtime      : 12.4% of planned time
  No-downtime OEE     : 80.3% vs 70.2% with downtime
  Top causes          : Maintenance · Breakdown · Material Shortage

DEFECTS:
  Reject rate         : 5.57% = 1.82M units wasted
  Top defect          : Scratch → tooling and surface finishing
  Defect rate varies  : significantly by machine and product

RECOMMENDATIONS:
1. Schedule PM in lowest-demand shift
2. Focus on bottom 10 machines for maintenance
3. Scratch reduction program — #1 defect type
4. Shift knowledge transfer from best to worst shift
5. Energy benchmarking across machines
6. Deploy ML OEE prediction for early warning
"""
    col1, col2, col3 = st.columns(3)
with col1:
    st.download_button("📥 Download Report (.txt)", report_txt,
                       file_name="Manufacturing_Report_M3.txt",
                       mime="text/plain", use_container_width=True)
with col2:
    machine_summary = df.groupby("machine_id").agg(
        Avg_OEE=("OEE","mean"), Avg_Downtime=("downtime_min","mean"),
        Avg_RejectRate=("reject_rate","mean"), Total_Shifts=("OEE","count"),
        Low_OEE_Shifts=("low_OEE","sum")
    ).round(4).sort_values("Avg_OEE",ascending=False).reset_index()
    st.download_button("📥 Machine Rankings (.csv)",
                       machine_summary.to_csv(index=False),
                       file_name="Machine_Performance_Rankings_M3.csv",
                       mime="text/csv", use_container_width=True)
with col3:
    downtime_summary = df.groupby("downtime_reason").agg(
        Shifts=("OEE","count"), Avg_OEE=("OEE","mean"),
        Total_Downtime=("downtime_min","sum")
    ).round(2).sort_values("Total_Downtime",ascending=False).reset_index()
    st.download_button("📥 Downtime KPIs (.csv)",
                       downtime_summary.to_csv(index=False),
                       file_name="Downtime_KPIs_M3.csv",
                       mime="text/csv", use_container_width=True)
