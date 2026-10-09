"""TalentLens 2026 — Employee Turnover Prediction Studio.
Premium Streamlit UI: hero dashboard, single + batch prediction, model insights.
Models: Decision Tree / Random Forest (scikit-learn), trained via train.py.
Run: streamlit run app.py
"""
from pathlib import Path
import json

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "employee_attrition_hr_2026.csv"
MODEL_DIR = BASE / "models"

st.set_page_config(page_title="TalentLens 2026 · Turnover Prediction",
                   page_icon="🧲", layout="wide",
                   initial_sidebar_state="expanded")

# ---------- theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:radial-gradient(1200px 600px at 15% -5%,#1b2a4a 0%,#0d1117 55%) fixed,#0d1117;color:#e6edf3;}
.hero{background:linear-gradient(135deg,#6e56cf 0%,#2563eb 45%,#06b6d4 100%);
  border-radius:20px;padding:34px 36px;color:white;box-shadow:0 20px 60px rgba(37,99,235,.35);margin-bottom:18px;}
.hero h1{font-size:2.4rem;font-weight:800;margin:0 0 6px;}
.hero p{opacity:.92;font-size:1.02rem;margin:4px 0;}
.kpi{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.12);border-radius:16px;
  padding:16px 18px;backdrop-filter:blur(8px);}
.kpi h3{margin:0;font-size:.8rem;letter-spacing:.08em;text-transform:uppercase;opacity:.75;}
.kpi h2{margin:4px 0 0;font-size:1.7rem;font-weight:800;}
.card{background:#161b22;border:1px solid #2a3340;border-radius:16px;padding:20px 22px;margin-bottom:16px;}
.risk-high{background:linear-gradient(135deg,#ef4444,#b91c1c);color:#fff;border-radius:14px;padding:18px;font-weight:700;}
.risk-med{background:linear-gradient(135deg,#f59e0b,#b45309);color:#fff;border-radius:14px;padding:18px;font-weight:700;}
.risk-low{background:linear-gradient(135deg,#10b981,#047857);color:#fff;border-radius:14px;padding:18px;font-weight:700;}
.stButton>button{background:linear-gradient(135deg,#6e56cf,#2563eb);color:#fff;border:0;border-radius:12px;
  padding:.65rem 1.4rem;font-weight:700;box-shadow:0 8px 24px rgba(37,99,235,.4);}
.stButton>button:hover{filter:brightness(1.1);}
section[data-testid="stSidebar"]{background:#0f141b;border-right:1px solid #232b36;}
footer{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

ROLE_BY_DEPT = {
    "Sales": ["Sales Executive", "Sales Manager", "Account Executive"],
    "Research & Development": ["Research Scientist", "Lab Technician", "R&D Manager"],
    "Human Resources": ["HR Business Partner", "HR Executive"],
    "Engineering": ["Software Engineer", "Data Scientist", "Engineering Manager"],
    "Marketing": ["Marketing Specialist", "Marketing Manager"],
    "Finance": ["Financial Analyst", "Finance Manager"],
}

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df

@st.cache_resource
def load_models():
    dt = joblib.load(MODEL_DIR / "attrition_dt.pkl")
    rf = joblib.load(MODEL_DIR / "attrition_rf.pkl")
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return {"Decision Tree": dt, "Random Forest": rf}, metrics

try:
    df = load_data()
    MODELS, METRICS = load_models()
except Exception as e:
    st.error(f"Couldn't load data/models. Run `python train.py` first. ({e})")
    st.stop()

# ---------- sidebar ----------
with st.sidebar:
    st.markdown("## 🧲 TalentLens 2026")
    st.caption("Modern turnover intelligence · DT + RF")
    page = st.radio("Navigate", ["🏠 Overview", "🔮 Predict one employee",
                                 "📤 Batch scoring", "📊 Model insights"], index=0)
    st.divider()
    model_name = st.selectbox("Model", ["Random Forest", "Decision Tree"],
                              help="Random Forest is the champion (AUC ~0.82).")
    threshold = st.slider("Alert threshold", 0.10, 0.90, 0.50, 0.05)
    st.caption("Flag as *likely to leave* when probability ≥ threshold.")
    st.divider()
    st.caption("Dataset: Kaggle · Employee Attrition & HR Analytics 2026 (5,000 rows, 29 cols)")

model = MODELS[model_name]

def risk_band(p):
    if p >= 0.65: return "HIGH", "risk-high", "🚨 Immediate retention action recommended."
    if p >= 0.35: return "MEDIUM", "risk-med", "⚠️ Monitor closely · 1:1 + growth plan."
    return "LOW", "risk-low", "✅ Healthy · keep recognising & growing."

def drivers(row: dict):
    """Human-readable risk drivers (heuristic, aligned with RF importances)."""
    d = []
    if row["BurnoutScore"] >= 4: d.append("🔥 Burnout score high (4–5)")
    if row["OvertimeHours"] >= 15: d.append(f"⏰ Heavy overtime ({row['OvertimeHours']}h/mo)")
    if row["EngagementScore"] <= 2: d.append("💔 Engagement low (1–2)")
    if row["JobSatisfaction"] <= 2: d.append("😞 Job satisfaction low")
    if row["YearsSinceLastPromotion"] >= 4: d.append(f"📉 Promotion drought ({row['YearsSinceLastPromotion']}y)")
    if row["ManagerSupport"] <= 2: d.append("🧊 Weak manager support")
    if row["AIJobRisk"] == "High": d.append("🤖 High perceived AI job-risk")
    if row["WorkLifeBalance"] <= 2: d.append("⚖️ Poor work-life balance")
    if row["CommuteMinutes"] >= 45 and row["WorkMode"] == "Onsite": d.append("🚗 Long onsite commute")
    return d if d else ["🌿 No major risk signals detected."]

# ================= OVERVIEW =================
if page == "🏠 Overview":
    st.markdown("""<div class="hero">
    <h1>🧲 TalentLens 2026 — who quits, and why?</h1>
    <p>Modern turnover prediction for the hybrid + AI-anxious workplace. Burnout, overtime,
    promotion droughts, manager support, work-mode & AI job-risk — modelled with
    <b>Decision Trees & Random Forests</b>.</p>
    <p>📦 5,000 employees · 29 features · champion <b>Random Forest · ROC-AUC 0.82</b></p></div>""",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    attr_rate = (df.Attrition == "Yes").mean()
    with c1: st.markdown(f'<div class="kpi"><h3>Attrition rate</h3><h2>{attr_rate:.1%}</h2></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi"><h3>Champion AUC</h3><h2>{METRICS["random_forest"]["roc_auc"]:.2f}</h2></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi"><h3>Avg burnout (left)</h3><h2>{df[df.Attrition=="Yes"].BurnoutScore.mean():.1f} / 5</h2></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi"><h3>Avg overtime (left)</h3><h2>{df[df.Attrition=="Yes"].OvertimeHours.mean():.0f}h</h2></div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Attrition by work mode — does flexibility retain?")
    g = df.groupby(["WorkMode", "Attrition"]).size().reset_index(name="n")
    fig = px.bar(g, x="WorkMode", y="n", color="Attrition", barmode="group",
                 color_discrete_map={"Yes": "#ef4444", "No": "#22c55e"},
                 title="Headcount by work mode")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e6edf3")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Burnout vs attrition")
        g2 = df.groupby(["BurnoutScore", "Attrition"]).size().reset_index(name="n")
        fig2 = px.bar(g2, x="BurnoutScore", y="n", color="Attrition", barmode="stack",
                      color_discrete_map={"Yes": "#ef4444", "No": "#3b82f6"})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Top risk drivers (Random Forest)")
        top = pd.DataFrame(METRICS["top_features"])
        fig3 = px.bar(top.iloc[::-1], x="importance", y="feature", orientation="h",
                      color="importance", color_continuous_scale="Blues")
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
                           coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.info("👉 Go to **🔮 Predict one employee** to score someone, or **📤 Batch scoring** for a whole CSV.")

# ================= SINGLE PREDICT =================
elif page == "🔮 Predict one employee":
    st.markdown('<div class="hero"><h1>🔮 Score an employee</h1><p>Live inference · probability of leaving + plain-English drivers + retention playbook.</p></div>',
                unsafe_allow_html=True)
    with st.form("score"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**👤 Profile**")
            Age = st.slider("Age", 21, 60, 32)
            Gender = st.selectbox("Gender", ["Male", "Female"])
            MaritalStatus = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
            Department = st.selectbox("Department", list(ROLE_BY_DEPT))
            JobRole = st.selectbox("Job role", ROLE_BY_DEPT[Department])
            JobLevel = st.slider("Job level", 1, 5, 2)
        with c2:
            st.markdown("**💼 Work**")
            MonthlyIncome = st.number_input("Monthly income ($)", 2500, 22000, 7000, 100)
            StockOptionLevel = st.selectbox("Stock options", [0, 1, 2, 3])
            PercentSalaryHike = st.slider("Last hike (%)", 5.0, 25.0, 13.0)
            TotalWorkingYears = st.slider("Total experience (y)", 0, 40, 8)
            YearsAtCompany = st.slider("Years at company", 0, 30, 4)
            YearsInCurrentRole = st.slider("Years in role", 0, 15, 2)
            YearsSinceLastPromotion = st.slider("Years since promotion", 0, 12, 2)
            NumCompaniesWorked = st.slider("Companies worked", 0, 9, 2)
            WorkMode = st.selectbox("Work mode", ["Onsite", "Hybrid", "Remote"])
            CommuteMinutes = st.slider("Commute (min)", 0, 90, 30)
            OvertimeHours = st.slider("Overtime (h/mo)", 0, 30, 8)
            BusinessTravel = st.selectbox("Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
            TrainingHoursLastYear = st.slider("Training (h/yr)", 0, 80, 25)
        with c3:
            st.markdown("**💚 Wellbeing + AI era**")
            ManagerSupport = st.slider("Manager support (1-5)", 1, 5, 3)
            BurnoutScore = st.slider("Burnout (1-5)", 1, 5, 3)
            EngagementScore = st.slider("Engagement (1-5)", 1, 5, 3)
            JobSatisfaction = st.slider("Job satisfaction (1-4)", 1, 4, 3)
            WorkLifeBalance = st.slider("Work-life balance (1-4)", 1, 4, 3)
            PerformanceRating = st.slider("Performance (1-4)", 1, 4, 3)
            AIToolsUsageHours = st.slider("AI-tool use (h/wk)", 0.0, 20.0, 5.0)
            AIJobRisk = st.selectbox("Perceived AI job-risk", ["Low", "Medium", "High"])
        go_btn = st.form_submit_button("⚡ Predict turnover risk")

    if go_btn:
        row = dict(Age=Age, Gender=Gender, MaritalStatus=MaritalStatus, Department=Department,
                   JobRole=JobRole, JobLevel=JobLevel, MonthlyIncome=MonthlyIncome,
                   StockOptionLevel=StockOptionLevel, PercentSalaryHike=PercentSalaryHike,
                   TotalWorkingYears=TotalWorkingYears, YearsAtCompany=YearsAtCompany,
                   YearsInCurrentRole=YearsInCurrentRole, YearsSinceLastPromotion=YearsSinceLastPromotion,
                   NumCompaniesWorked=NumCompaniesWorked, WorkMode=WorkMode, CommuteMinutes=CommuteMinutes,
                   OvertimeHours=OvertimeHours, BusinessTravel=BusinessTravel,
                   TrainingHoursLastYear=TrainingHoursLastYear, ManagerSupport=ManagerSupport,
                   BurnoutScore=BurnoutScore, EngagementScore=EngagementScore,
                   JobSatisfaction=JobSatisfaction, WorkLifeBalance=WorkLifeBalance,
                   PerformanceRating=PerformanceRating, AIToolsUsageHours=AIToolsUsageHours, AIJobRisk=AIJobRisk)
        X = pd.DataFrame([row])
        p = float(model.predict_proba(X)[0, 1])
        band, css, msg = risk_band(p)
        flagged = p >= threshold
        st.markdown(f'<div class="{css}">Risk: {band} · {p:.1%} probability of leaving &nbsp;|&nbsp; '
                    f'{"🚩 FLAGGED" if flagged else "not flagged"} (threshold {threshold:.0%})<br>{msg}</div>',
                    unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=p * 100,
                                       title={"text": "Attrition probability (%)"},
                                       gauge={"axis": {"range": [0, 100]},
                                              "bar": {"color": "#ef4444" if band == "HIGH" else "#f59e0b" if band == "MEDIUM" else "#10b981"},
                                              "steps": [{"range": [0, 35], "color": "#064e3b"},
                                                        {"range": [35, 65], "color": "#78350f"},
                                                        {"range": [65, 100], "color": "#7f1d1d"}]}))
        gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3", height=280)
        c1, c2 = st.columns([1, 1])
        with c1: st.plotly_chart(gauge, use_container_width=True)
        with c2:
            st.markdown('<div class="card"><b>🔍 Why this score?</b><ul>' +
                        "".join(f"<li>{x}</li>" for x in drivers(row)) + "</ul></div>", unsafe_allow_html=True)
            st.markdown('<div class="card"><b>💡 Retention playbook</b><ul>'
                        "<li>Burnout ≥4 → cut overtime, 1:1 this week, 2 days WFH</li>"
                        "<li>Promotion drought ≥4y → promotion / role-expansion review</li>"
                        "<li>AI-risk High → reskill plan + AI-tool training hours</li>"
                        "<li>Manager ≤2 → switch mentor / feedback loop</li></ul></div>", unsafe_allow_html=True)

# ================= BATCH =================
elif page == "📤 Batch scoring":
    st.markdown('<div class="hero"><h1>📤 Batch scoring</h1><p>Upload any CSV with the same 27 feature columns — get probabilities + flags back.</p></div>',
                unsafe_allow_html=True)
    st.download_button("⬇️ Download template (first 5 rows)",
                       df.drop(columns=["Attrition", "EmployeeID"]).head().to_csv(index=False),
                       "template.csv", "text/csv")
    f = st.file_uploader("Drop employee CSV here", type="csv")
    if f:
        bdf = pd.read_csv(f)
        need = [c for c in df.drop(columns=["Attrition", "EmployeeID"]).columns if c not in bdf.columns]
        if need:
            st.error(f"Missing columns: {need}")
        else:
            with st.spinner("Scoring…"):
                proba = model.predict_proba(bdf[df.drop(columns=["Attrition", "EmployeeID"]).columns])[:, 1]
                out = bdf.copy()
                out["AttritionProb"] = (proba * 100).round(1)
                out["Prediction"] = ["Yes" if p >= threshold else "No" for p in proba]
                out["RiskBand"] = pd.cut(proba, [0, 0.35, 0.65, 1.01], labels=["Low", "Medium", "High"])
            st.success(f"Scored {len(out)} employees · {sum(out.Prediction=='Yes')} flagged.")
            st.dataframe(out.head(20), use_container_width=True)
            fig = px.histogram(out, x="AttritionProb", nbins=25, title="Risk distribution",
                               color="RiskBand", color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Download scored CSV", out.to_csv(index=False), "scored.csv", "text/csv")

# ================= INSIGHTS =================
else:
    st.markdown('<div class="hero"><h1>📊 Model insights</h1><p>Champion vs challenger · what the forest learned · where the data says to act.</p></div>',
                unsafe_allow_html=True)
    mdt, mrf = METRICS["decision_tree"], METRICS["random_forest"]
    t = pd.DataFrame([{"Model": "Decision Tree", **mdt}, {"Model": "Random Forest", **mrf}])
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.dataframe(t[["Model", "accuracy", "precision", "recall", "f1", "roc_auc"]].style.format("{:.3f}", subset=["accuracy", "precision", "recall", "f1", "roc_auc"]),
                 use_container_width=True)
    st.caption(f"Best params — DT: {METRICS['best_params']['dt']} · RF: {METRICS['best_params']['rf']} · test n={METRICS['test_size']}")
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><b>Confusion — Random Forest</b></div>', unsafe_allow_html=True)
        cm = mrf["confusion_matrix"]
        fig = px.imshow(cm, text_auto=True, x=["Stayed", "Left"], y=["Stayed", "Left"],
                        color_continuous_scale="Blues", title="RF confusion matrix")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown('<div class="card"><b>Metric face-off</b></div>', unsafe_allow_html=True)
        melt = t.melt(id_vars="Model", value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
                      var_name="metric", value_name="score")
        fig = px.bar(melt, x="metric", y="score", color="Model", barmode="group",
                     color_discrete_map={"Decision Tree": "#8b5cf6", "Random Forest": "#06b6d4"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
                          yaxis_range=[0.4, 1])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Promotion drought — where is the cliff?")
    g = df.groupby("YearsSinceLastPromotion")["Attrition"].apply(lambda s: (s == "Yes").mean()).reset_index(name="rate")
    fig = px.line(g, x="YearsSinceLastPromotion", y="rate", markers=True, title="Attrition rate vs years since promotion")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<center style='opacity:.55'>TalentLens 2026 · DT + RF · scikit-learn · Streamlit · built for ML Lab mini-project</center>",
            unsafe_allow_html=True)
