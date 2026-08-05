"""
Repo_12_Industrial_Manufacturing — ML_Models.py  (5 Tabs)
Author : Mohamed · M3
Regression     → OEE (Overall Equipment Effectiveness)
Classification → low_OEE (1 if OEE < 0.65)  class_weight='balanced'
"""
import os, pathlib, warnings, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import psutil

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model      import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree              import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble          import (RandomForestRegressor, GradientBoostingRegressor,
                                       RandomForestClassifier, GradientBoostingClassifier)
from sklearn.svm               import LinearSVC
from sklearn.calibration       import CalibratedClassifierCV
from sklearn.neighbors         import KNeighborsClassifier

warnings.filterwarnings("ignore")
S = st.session_state

st.set_page_config(page_title="ML Models · Industrial Manufacturing · M3",
                   page_icon="🤖", layout="wide")

LOGO = pathlib.Path(__file__).parent.parent / "M3_logo.png"

_data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "manufacturing_clean.csv"
)

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🤖 ML Models")
    st.markdown("Industrial Manufacturing · 5 Tabs")
    st.divider()
    st.markdown("### 📂 Dataset")
    st.success("✅ manufacturing_clean.csv")
    st.divider()
    st.markdown("### ⚙️ Options")
    test_size    = st.slider("Test Split %", 10, 40, 20, 5) / 100
    use_parallel = st.checkbox("Parallel (n_jobs=-1)", value=True)
    n_jobs       = -1 if use_parallel else 1
    st.info("OEE = Availability × Performance × Quality\n\nLow OEE rate: 30.9%\nclass_weight='balanced'")

CLR = {"primary":"#1565c0","success":"#2e7d32","warning":"#e65100",
       "danger":"#c62828","teal":"#00695c","light":"#e3f2fd","dark":"#1a237e",
       "amber":"#f57f17","grey":"#546e7a"}

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0f1923;}
[data-testid="stSidebar"] *{color:#e0e8f0 !important;}
.main{background:#f4f7fb;}
div[data-testid="metric-container"]{background:#e8f5e9;border-left:4px solid #00695c;border-radius:6px;padding:10px 14px;}
.sec-header{background:linear-gradient(90deg,#00695c,#1565c0);color:#ffffff !important;
  padding:10px 18px;border-radius:8px;font-size:1.1rem;font-weight:700;margin-bottom:16px;}
.insight-box{background:#e8f5e9;border-left:4px solid #2e7d32;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.insight-box p{color:#1b3a1f !important;margin:0;font-size:0.93rem;}
.warn-box{background:#fff3e0;border-left:4px solid #e65100;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.warn-box p{color:#4a2000 !important;margin:0;font-size:0.93rem;}
.info-box{background:#e3f2fd;border-left:4px solid #1565c0;padding:12px 16px;border-radius:0 6px 6px 0;margin:8px 0;}
.info-box p{color:#0d2a4a !important;margin:0;font-size:0.93rem;}
</style>""", unsafe_allow_html=True)

def sec(t): st.markdown(f'<div class="sec-header">{t}</div>', unsafe_allow_html=True)
def insight(t): st.markdown(f'<div class="insight-box"><p>✅ {t}</p></div>', unsafe_allow_html=True)
def warn(t):    st.markdown(f'<div class="warn-box"><p>⚠️ {t}</p></div>', unsafe_allow_html=True)
def info(t):    st.markdown(f'<div class="info-box"><p>ℹ️ {t}</p></div>', unsafe_allow_html=True)

def get_cpu_info(use_parallel, n_jobs):
    return {"total": os.cpu_count(), "used": n_jobs if use_parallel else 1,
            "percent": psutil.cpu_percent(interval=0.3)}

# ── LOAD ─────────────────────────────────────────────────────
if not os.path.exists(_data_path):
    st.error(f"❌ File not found: {_data_path}")
    st.info("Run P12_clean_data.py in Jupyter → copy manufacturing_clean.csv to data/ folder")
    st.stop()

df = pd.read_csv(_data_path, sep=",", decimal=".")
df.columns = df.columns.str.strip()
if df.empty:
    st.warning("⚠️ Dataset is empty."); st.stop()

S["df_work"] = df

REG_TARGET = "OEE"
CLF_TARGET = "low_OEE"
low_rate   = df[CLF_TARGET].mean() * 100

# ── FEATURES ─────────────────────────────────────────────────
FEATURE_COLS = [c for c in [
    "availability","performance","quality",
    "downtime_min","downtime_rate","operating_time_min",
    "total_units","good_units","reject_units","reject_rate",
    "energy_kwh","energy_per_unit","production_efficiency",
    "no_downtime","has_breakdown","has_defect",
    "shift_enc","machine_id_enc","product_id_enc","operator_id_enc",
    "downtime_reason_enc","defect_type_enc",
    "month","day_of_week","is_weekend"
] if c in df.columns]

df_ml = df[FEATURE_COLS + [REG_TARGET, CLF_TARGET]].dropna().copy()
X     = df_ml[FEATURE_COLS]
y_reg = df_ml[REG_TARGET]
y_clf = df_ml[CLF_TARGET]

X_tr_r, X_te_r, yr_tr, yr_te = train_test_split(
    X, y_reg, test_size=test_size, random_state=42)
X_tr_c, X_te_c, yc_tr, yc_te = train_test_split(
    X, y_clf, test_size=test_size, random_state=42, stratify=y_clf)

scaler   = StandardScaler()
Xtr_r_sc = scaler.fit_transform(X_tr_r)
Xte_r_sc = scaler.transform(X_te_r)
Xtr_c_sc = scaler.fit_transform(X_tr_c)
Xte_c_sc = scaler.transform(X_te_c)

REG_MODELS = {
    "Linear Regression": LinearRegression(),
    "Ridge":             Ridge(alpha=1.0),
    "Lasso":             Lasso(alpha=0.001, max_iter=5000),
    "Decision Tree":     DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}
CLF_MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced",
                                               n_jobs=n_jobs, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(max_depth=8, class_weight="balanced",
                                                   random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                   n_jobs=n_jobs, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM (Linear)":        CalibratedClassifierCV(
                               LinearSVC(class_weight="balanced", max_iter=3000, random_state=42)),
    "KNN":                 KNeighborsClassifier(n_neighbors=7, n_jobs=n_jobs),
}

# ── TABS ─────────────────────────────────────────────────────
tabs = st.tabs(["1 · Model Training",
                "2 · Regression Results",
                "3 · Classification Results",
                "4 · Feature Importance",
                "5 · Predict OEE"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    sec("🚀 Tab 1 — Model Training")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Shift Records",  f"{len(df_ml):,}")
    c2.metric("Features",       f"{len(FEATURE_COLS)}")
    c3.metric("Train Size",     f"{len(X_tr_r):,}")
    c4.metric("Test Size",      f"{len(X_te_r):,}")
    c5.metric("Low OEE Rate",   f"{low_rate:.1f}%")

    cpu = get_cpu_info(use_parallel, n_jobs)
    st.info(f"🖥 CPU: {cpu['total']} cores · Using: {cpu['used']} · Load: {cpu['percent']}%")

    col1, col2 = st.columns(2)
    with col1:
        sec("📈 Regression Target")
        st.markdown(f"**`{REG_TARGET}`** — Overall Equipment Effectiveness (0–1)")
        st.markdown(f"Mean={y_reg.mean():.3f} · Range={y_reg.min():.3f}–{y_reg.max():.3f}")
    with col2:
        sec("🎯 Classification Target")
        st.markdown(f"**`{CLF_TARGET}`** — 1 if OEE < 65% (poor performance)")
        st.markdown(f"Low OEE rate={low_rate:.1f}% → class_weight='balanced'")

    if "reg_results" not in S: S["reg_results"] = []
    if "reg_models"  not in S: S["reg_models"]  = {}
    if "clf_results" not in S: S["clf_results"] = []
    if "clf_models"  not in S: S["clf_models"]  = {}
    S["X_te_r"] = X_te_r; S["Xte_r_sc"] = Xte_r_sc
    S["X_te_c"] = X_te_c; S["Xte_c_sc"] = Xte_c_sc
    S["yr_te"]  = yr_te;  S["yc_te"]    = yc_te
    S["scaler"] = scaler; S["X_cols"]   = FEATURE_COLS

    def _done_r(n): return any(r["Model"]==n for r in S["reg_results"])
    def _done_c(n): return any(r["Model"]==n for r in S["clf_results"])

    def _train_reg(name, model):
        use_sc = name in ["Linear Regression","Ridge","Lasso"]
        Xtr = Xtr_r_sc if use_sc else X_tr_r
        Xte = Xte_r_sc if use_sc else X_te_r
        t0  = time.time(); model.fit(Xtr, yr_tr); preds = model.predict(Xte)
        row = {"Model":name,
               "R²":   round(r2_score(yr_te, preds),4),
               "MAE":  round(mean_absolute_error(yr_te, preds),4),
               "RMSE": round(np.sqrt(mean_squared_error(yr_te, preds)),4),
               "Time(s)": round(time.time()-t0,2)}
        S["reg_results"] = [r for r in S["reg_results"] if r["Model"]!=name] + [row]
        S["reg_models"][name] = model
        return row

    def _train_clf(name, model):
        use_sc = name in ["Logistic Regression","SVM (Linear)","KNN"]
        Xtr = Xtr_c_sc if use_sc else X_tr_c
        Xte = Xte_c_sc if use_sc else X_te_c
        t0  = time.time(); model.fit(Xtr, yc_tr); preds = model.predict(Xte)
        proba = model.predict_proba(Xte)[:,1] if hasattr(model,"predict_proba") else None
        row = {"Model":name,
               "F1":        round(f1_score(yc_te, preds, zero_division=0),4),
               "Recall":    round(recall_score(yc_te, preds, zero_division=0),4),
               "Precision": round(precision_score(yc_te, preds, zero_division=0),4),
               "Accuracy":  round(accuracy_score(yc_te, preds),4),
               "ROC-AUC":   round(roc_auc_score(yc_te,proba),4) if proba is not None else 0.0,
               "Time(s)":   round(time.time()-t0,2)}
        S["clf_results"] = [r for r in S["clf_results"] if r["Model"]!=name] + [row]
        S["clf_models"][name] = model
        return row

    st.markdown("---")
    sec("📈 Regression Models — Train Individually")
    rc = st.columns(3)
    for i,(name,model) in enumerate(REG_MODELS.items()):
        with rc[i%3]:
            label = f"✅ {name}" if _done_r(name) else f"▶ Train {name}"
            if st.button(label, key=f"reg_{name}", use_container_width=True):
                with st.spinner(f"Training {name}..."):
                    row = _train_reg(name, model)
                st.success(f"R²={row['R²']:.4f} · MAE={row['MAE']:.4f} · {row['Time(s)']}s")
                st.rerun()
            if _done_r(name):
                r = next(r for r in S["reg_results"] if r["Model"]==name)
                st.caption(f"R²={r['R²']:.4f} · MAE={r['MAE']:.4f}")

    if S["reg_results"]:
        st.dataframe(pd.DataFrame(S["reg_results"]).sort_values("R²",ascending=False)
                       .reset_index(drop=True)
                       .style.background_gradient(subset=["R²"],cmap="RdYlGn")
                       .format({"R²":"{:.4f}","MAE":"{:.4f}","RMSE":"{:.4f}"}),
                     use_container_width=True)

    st.markdown("---")
    sec("🎯 Classification Models — Train Individually")
    warn("Low OEE rate 30.9% → class_weight='balanced' on all models")
    cc = st.columns(3)
    for i,(name,model) in enumerate(CLF_MODELS.items()):
        with cc[i%3]:
            label = f"✅ {name}" if _done_c(name) else f"▶ Train {name}"
            if st.button(label, key=f"clf_{name}", use_container_width=True):
                with st.spinner(f"Training {name}..."):
                    row = _train_clf(name, model)
                st.success(f"F1={row['F1']:.4f} · AUC={row['ROC-AUC']:.4f} · {row['Time(s)']}s")
                st.rerun()
            if _done_c(name):
                r = next(r for r in S["clf_results"] if r["Model"]==name)
                st.caption(f"F1={r['F1']:.4f} · AUC={r['ROC-AUC']:.4f}")

    if S["clf_results"]:
        st.dataframe(pd.DataFrame(S["clf_results"]).sort_values("F1",ascending=False)
                       .reset_index(drop=True)
                       .style.background_gradient(subset=["F1","ROC-AUC"],cmap="RdYlGn")
                       .format({c:"{:.4f}" for c in ["F1","Recall","Precision","Accuracy","ROC-AUC"]}),
                     use_container_width=True)

    n_done = len(S["reg_results"]) + len(S["clf_results"])
    st.info(f"📊 {n_done}/12 models trained." if n_done < 12
            else "✅ All 12 models trained! Navigate to Results tabs →")

# ══════════════════════════════════════════════════════════════
# TAB 2 — REGRESSION RESULTS
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    sec("📈 Tab 2 — Regression Results")
    info("Predicting: **OEE** — Overall Equipment Effectiveness (0–1)")

    if not S.get("reg_results"):
        warn("Train at least one Regression model in Tab 1.")
    else:
        reg_df   = pd.DataFrame(S["reg_results"]).sort_values("R²",ascending=False).reset_index(drop=True)
        best_reg = reg_df.iloc[0]["Model"]

        st.dataframe(reg_df.style.background_gradient(subset=["R²"],cmap="RdYlGn")
                                  .background_gradient(subset=["MAE","RMSE"],cmap="RdYlGn_r")
                                  .format({"R²":"{:.4f}","MAE":"{:.4f}","RMSE":"{:.4f}"}),
                     use_container_width=True)
        st.markdown(f"🏆 **Best:** `{best_reg}` — R²={reg_df.iloc[0]['R²']:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(reg_df, x="Model", y="R²",
                         color="R²",
                         color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                         title="R² — All Regression Models",
                         text=reg_df["R²"].apply(lambda x: f"{x:.4f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=370, xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="MAE",  x=reg_df["Model"],
                                  y=reg_df["MAE"],  marker_color=CLR["warning"]))
            fig2.add_trace(go.Bar(name="RMSE", x=reg_df["Model"],
                                  y=reg_df["RMSE"], marker_color=CLR["danger"]))
            fig2.update_layout(barmode="group", height=370,
                                title="MAE vs RMSE", xaxis_tickangle=-25)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        sec(f"📈 Actual vs Predicted — {best_reg}")
        bm   = S["reg_models"][best_reg]
        Xte  = S["Xte_r_sc"] if best_reg in ["Linear Regression","Ridge","Lasso"] else S["X_te_r"]
        pred = bm.predict(Xte)

        col3, col4 = st.columns(2)
        with col3:
            fig3, ax = plt.subplots(figsize=(7,5))
            ax.scatter(yr_te, pred, alpha=0.2, s=5, color=CLR["teal"])
            lims = [min(yr_te.min(),pred.min()), max(yr_te.max(),pred.max())]
            ax.plot(lims,lims,"r--",lw=2,label="Perfect fit")
            ax.axhline(0.85, color=CLR["success"], ls=":", lw=1.5, label="World class")
            ax.set_xlabel("Actual OEE"); ax.set_ylabel("Predicted OEE")
            ax.set_title(f"Actual vs Predicted — {best_reg}"); ax.legend()
            plt.tight_layout(); st.pyplot(fig3); plt.close()
        with col4:
            resid = yr_te.values - pred
            fig4, ax2 = plt.subplots(figsize=(7,5))
            ax2.scatter(pred, resid, alpha=0.2, s=5, color=CLR["amber"])
            ax2.axhline(0, color=CLR["danger"], lw=2, ls="--")
            ax2.set_xlabel("Predicted OEE"); ax2.set_ylabel("Residual")
            ax2.set_title("Residual Plot")
            plt.tight_layout(); st.pyplot(fig4); plt.close()

        info("High R² expected — OEE is mathematically derived from availability×performance×quality features.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — CLASSIFICATION RESULTS
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    sec("🎯 Tab 3 — Classification Results")
    info("Predicting: **low_OEE** — 1 if shift has poor OEE < 65%")

    if not S.get("clf_results"):
        warn("Train at least one Classification model in Tab 1.")
    else:
        clf_df   = pd.DataFrame(S["clf_results"]).sort_values("F1",ascending=False).reset_index(drop=True)
        best_clf = clf_df.iloc[0]["Model"]
        yc_te_s  = S["yc_te"]

        st.dataframe(clf_df.style.background_gradient(subset=["F1","ROC-AUC"],cmap="RdYlGn")
                                  .format({c:"{:.4f}" for c in ["F1","Recall","Precision","Accuracy","ROC-AUC"]}),
                     use_container_width=True)
        st.markdown(f"🏆 **Best:** `{best_clf}` — F1={clf_df.iloc[0]['F1']:.4f} · AUC={clf_df.iloc[0]['ROC-AUC']:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(clf_df, x="Model", y="F1",
                         color="F1",
                         color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                         title="F1 Score — All Classifiers",
                         text=clf_df["F1"].apply(lambda x: f"{x:.4f}"))
            fig.update_traces(textposition="outside")
            fig.update_layout(height=370, xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(clf_df, x="Model", y="ROC-AUC",
                          color="ROC-AUC",
                          color_continuous_scale=["#c62828","#e65100","#2e7d32"],
                          title="ROC-AUC — All Classifiers",
                          text=clf_df["ROC-AUC"].apply(lambda x: f"{x:.4f}"))
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=370, xaxis_tickangle=-25)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)
        bm      = S["clf_models"][best_clf]
        use_sc  = best_clf in ["Logistic Regression","SVM (Linear)","KNN"]
        Xte_c   = S["Xte_c_sc"] if use_sc else S["X_te_c"]
        preds_c = bm.predict(Xte_c)
        cm      = confusion_matrix(yc_te_s, preds_c)

        with col3:
            sec(f"🔢 Confusion Matrix — {best_clf}")
            fig3, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Normal OEE","Low OEE"],
                        yticklabels=["Normal OEE","Low OEE"], ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — {best_clf}")
            plt.tight_layout(); st.pyplot(fig3); plt.close()
            tn,fp,fn,tp = cm.ravel()
            st.markdown(f"**TP={tp:,}** correctly predicted low-OEE shifts · **FN={fn:,}** missed poor shifts")

        with col4:
            sec(f"📈 ROC Curve — {best_clf}")
            if hasattr(bm,"predict_proba"):
                proba_c = bm.predict_proba(Xte_c)[:,1]
                fpr,tpr,_ = roc_curve(yc_te_s, proba_c)
                auc_val   = roc_auc_score(yc_te_s, proba_c)
                fig4, ax2 = plt.subplots(figsize=(5,4))
                ax2.plot(fpr, tpr, color=CLR["teal"], lw=2.5,
                         label=f"AUC={auc_val:.4f}")
                ax2.plot([0,1],[0,1], color=CLR["grey"], ls="--")
                ax2.fill_between(fpr, tpr, alpha=0.1, color=CLR["teal"])
                ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
                ax2.set_title(f"ROC Curve — {best_clf}"); ax2.legend()
                plt.tight_layout(); st.pyplot(fig4); plt.close()

        insight(f"Best classifier: {best_clf} · catching poor-OEE shifts early enables proactive intervention.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    sec("🔑 Tab 4 — Feature Importance")

    if not S.get("clf_models"):
        warn("Train at least one model in Tab 1.")
    else:
        clf_df = pd.DataFrame(S["clf_results"]).sort_values("F1",ascending=False).reset_index(drop=True)
        reg_df = pd.DataFrame(S["reg_results"]).sort_values("R²",ascending=False).reset_index(drop=True) if S.get("reg_results") else pd.DataFrame()
        feats  = S["X_cols"]

        col1, col2 = st.columns(2)
        with col1:
            sec("🎯 Classification — Low OEE Predictors")
            best_clf = clf_df.iloc[0]["Model"]
            bm = S["clf_models"][best_clf]
            if hasattr(bm,"feature_importances_"):
                imp = pd.DataFrame({"Feature":feats,"Importance":bm.feature_importances_})\
                        .sort_values("Importance",ascending=True)
                fig, ax = plt.subplots(figsize=(7,max(5,len(imp)*0.35)))
                colors_i = [CLR["teal"] if i>=len(imp)-3 else CLR["primary"]
                            for i in range(len(imp))]
                ax.barh(imp["Feature"], imp["Importance"], color=colors_i)
                ax.set_xlabel("Importance")
                ax.set_title(f"{best_clf} — Low OEE Predictors")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            elif hasattr(bm,"coef_"):
                coef_vals = bm.coef_[0] if bm.coef_.ndim>1 else bm.coef_
                coef = pd.DataFrame({"Feature":feats,"Coef":np.abs(coef_vals)})\
                         .sort_values("Coef",ascending=True)
                fig, ax = plt.subplots(figsize=(7,max(5,len(coef)*0.35)))
                ax.barh(coef["Feature"], coef["Coef"], color=CLR["teal"])
                ax.set_xlabel("|Coefficient|"); ax.set_title(f"{best_clf}")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            else:
                info(f"{best_clf} doesn't expose feature importances.")

        with col2:
            sec("📈 Regression — OEE Predictors")
            if not reg_df.empty:
                best_reg = reg_df.iloc[0]["Model"]
                rm = S["reg_models"][best_reg]
                if hasattr(rm,"feature_importances_"):
                    imp2 = pd.DataFrame({"Feature":feats,"Importance":rm.feature_importances_})\
                             .sort_values("Importance",ascending=True)
                    fig2, ax2 = plt.subplots(figsize=(7,max(5,len(imp2)*0.35)))
                    ax2.barh(imp2["Feature"], imp2["Importance"], color=CLR["amber"])
                    ax2.set_xlabel("Importance")
                    ax2.set_title(f"{best_reg} — OEE Predictors")
                    plt.tight_layout(); st.pyplot(fig2); plt.close()
                elif hasattr(rm,"coef_"):
                    coef2 = pd.DataFrame({"Feature":feats,"Coef":np.abs(rm.coef_)})\
                              .sort_values("Coef",ascending=True)
                    fig2, ax2 = plt.subplots(figsize=(7,max(5,len(coef2)*0.35)))
                    ax2.barh(coef2["Feature"], coef2["Coef"], color=CLR["amber"])
                    ax2.set_xlabel("|Coefficient|"); ax2.set_title(f"{best_reg}")
                    plt.tight_layout(); st.pyplot(fig2); plt.close()
            else:
                info("Train regression models in Tab 1.")

        insight("availability, no_downtime, downtime_rate typically rank as top OEE predictors.")
        insight("reject_rate and quality rank high for classification — defect presence drives poor OEE.")

# ══════════════════════════════════════════════════════════════
# TAB 5 — PREDICT OEE
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    sec("🔮 Tab 5 — Interactive OEE Predictor")

    if not S.get("clf_models"):
        warn("Train at least one model in Tab 1.")
    else:
        info("Enter shift parameters to predict OEE and flag low-performance risk.")

        col1, col2, col3 = st.columns(3)
        with col1:
            sec("🏭 Shift Setup")
            shift_sel    = st.selectbox("Shift", ["Shift_1","Shift_2","Shift_3"])
            machine_id   = st.selectbox("Machine", [f"M{str(i).zfill(2)}" for i in range(1,51)])
            product_id   = st.selectbox("Product", ["P_A","P_B","P_C","P_D","P_E"])
            planned_time = 480  # always 480 min

        with col2:
            sec("⏱ Downtime & Production")
            downtime_min   = st.slider("Downtime (min)",    0, 300, 60, 5)
            total_units    = st.slider("Total Units",       100, 1000, 500, 10)
            reject_units   = st.slider("Reject Units",      0, 100, 25, 1)
            downtime_reason= st.selectbox("Downtime Reason",
                                          ["No Downtime","Maintenance","Breakdown",
                                           "Material Shortage","Setup"])

        with col3:
            sec("⚡ Energy & Derived")
            energy_kwh  = st.slider("Energy (kWh)", 100, 600, 280, 10)
            defect_type = st.selectbox("Defect Type",
                                       ["No Defect","Scratch","Dimension Error",
                                        "Crack","Surface Defect"])
            month_sel   = st.slider("Month", 1, 12, 6)
            dow_sel     = st.slider("Day of Week (0=Mon)", 0, 6, 1)

            # Computed
            operating_time = planned_time - downtime_min
            good_units     = total_units - reject_units
            avail          = round(operating_time / planned_time, 4)
            perf           = round(min(total_units / max(operating_time * 1.5, 1), 1.0), 4)
            qual           = round(good_units / max(total_units, 1), 4)
            oee_calc       = round(avail * perf * qual, 4)

            st.metric("Availability",    f"{avail:.1%}")
            st.metric("Quality",         f"{qual:.1%}")
            st.metric("Estimated OEE",   f"{oee_calc:.1%}",
                      delta=f"{'⚠️ Poor' if oee_calc < 0.65 else '✅ OK'}")

        st.markdown("---")
        if st.button("🔮 Predict OEE Risk", type="primary", use_container_width=True):
            # Encode categoricals using dataset mode
            def enc_cat(col, val):
                if col+"_enc" in df.columns and col in df.columns:
                    mapping = dict(zip(df[col].astype(str),
                                       df[col+"_enc"]))
                    return int(mapping.get(str(val), 0))
                return 0

            input_row = pd.DataFrame([{
                "availability":         avail,
                "performance":          perf,
                "quality":              qual,
                "downtime_min":         downtime_min,
                "downtime_rate":        round(downtime_min/planned_time, 4),
                "operating_time_min":   operating_time,
                "total_units":          total_units,
                "good_units":           good_units,
                "reject_units":         reject_units,
                "reject_rate":          round(reject_units/max(total_units,1), 4),
                "energy_kwh":           energy_kwh,
                "energy_per_unit":      round(energy_kwh/max(good_units,1), 4),
                "production_efficiency":qual,
                "no_downtime":          1 if downtime_min==0 else 0,
                "has_breakdown":        1 if downtime_reason=="Breakdown" else 0,
                "has_defect":           1 if defect_type!="No Defect" else 0,
                "shift_enc":            enc_cat("shift", shift_sel),
                "machine_id_enc":       enc_cat("machine_id", machine_id),
                "product_id_enc":       enc_cat("product_id", product_id),
                "operator_id_enc":      0,
                "downtime_reason_enc":  enc_cat("downtime_reason", downtime_reason),
                "defect_type_enc":      enc_cat("defect_type", defect_type),
                "month":                month_sel,
                "day_of_week":          dow_sel,
                "is_weekend":           1 if dow_sel >= 5 else 0,
            }])

            input_aligned = pd.DataFrame([{k: input_row[k].iloc[0]
                                           if k in input_row.columns else 0
                                           for k in S["X_cols"]}])
            input_sc = S["scaler"].transform(input_aligned)

            sec("🎯 Low OEE Risk — All Classifiers")
            pred_rows = []
            for name, model in S["clf_models"].items():
                use_sc = name in ["Logistic Regression","SVM (Linear)","KNN"]
                Xin    = input_sc if use_sc else input_aligned
                pred   = model.predict(Xin)[0]
                prob   = model.predict_proba(Xin)[0][1] if hasattr(model,"predict_proba") else None
                pred_rows.append({
                    "Model":       name,
                    "Prediction":  "⚠️ LOW OEE RISK" if pred==1 else "✅ NORMAL",
                    "Probability": f"{prob*100:.1f}%" if prob is not None else "N/A",
                })
            pred_df = pd.DataFrame(pred_rows)
            st.dataframe(pred_df, use_container_width=True)

            low_votes = sum(1 for r in pred_rows if "LOW" in r["Prediction"])
            if low_votes > len(pred_rows)/2:
                st.error(f"⚠️ LOW OEE RISK — {low_votes}/{len(pred_rows)} models flag this shift. Take action now.")
            else:
                st.success(f"✅ Shift expected to perform normally — only {low_votes}/{len(pred_rows)} models flag risk.")

            st.markdown("---")
            sec("📋 Shift Risk Factors")
            factors = []
            if downtime_min > 120:
                factors.append(("⚠️","High downtime: "+str(downtime_min)+" min — above 25% of planned time"))
            if downtime_reason == "Breakdown":
                factors.append(("🔴","Breakdown recorded — investigate root cause immediately"))
            if qual < 0.90:
                factors.append(("⚠️",f"Quality: {qual:.1%} — reject rate {reject_units/max(total_units,1):.1%} is high"))
            if defect_type not in ["No Defect"]:
                factors.append(("⚠️",f"Defect type: {defect_type} — inspect tooling and process parameters"))
            if oee_calc < 0.65:
                factors.append(("🔴",f"Estimated OEE {oee_calc:.1%} — below poor OEE threshold of 65%"))
            if energy_kwh/max(good_units,1) > 0.6:
                factors.append(("⚠️",f"Energy per unit: {energy_kwh/max(good_units,1):.3f} kWh — above average"))

            if factors:
                for level, msg in factors:
                    color = "warn" if "⚠️" in level else "info"
                    if "🔴" in level:
                        st.markdown(f'<div class="warn-box"><p>🔴 {msg}</p></div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="info-box"><p>⚠️ {msg}</p></div>',
                                    unsafe_allow_html=True)
            else:
                st.markdown('<div class="insight-box"><p>✅ No risk factors detected — shift parameters within normal range.</p></div>',
                            unsafe_allow_html=True)
