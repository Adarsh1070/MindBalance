import os
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -------------------- Page setup --------------------
st.set_page_config(
    page_title="MindBalance-AI | Student Wellness",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stress_predictor_model.pkl")

# -------------------- Styling --------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f6f8ff 0%, #f5fbff 55%, #f7f5ff 100%); }
    [data-testid="stHeader"] { background: rgba(246,248,255,.85); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14264d 0%, #1d3264 100%);
        border-right: 0;
    }
    [data-testid="stSidebar"] * { color: #f4f7ff; }
    [data-testid="stSidebar"] .stRadio label { padding: .35rem 0; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: #172554; }
    .hero {
        padding: 1.55rem 1.7rem; border-radius: 24px; color: #fff;
        background: linear-gradient(115deg, #243d8f 0%, #3e62d8 52%, #13a6a6 100%);
        box-shadow: 0 12px 32px rgba(49, 75, 170, .18); margin-bottom: 1.1rem;
    }
    .hero h1 { color: white !important; font-size: clamp(1.8rem, 3vw, 2.55rem); margin: 0 0 .4rem 0; }
    .hero p { color: #e9efff; margin: 0; font-size: 1.02rem; }
    .eyebrow { text-transform: uppercase; letter-spacing: .13em; font-size: .72rem; font-weight: 700; opacity: .82; }
    .panel {
        background: rgba(255,255,255,.91); border: 1px solid #e4e9ff;
        border-radius: 20px; padding: 1.1rem 1.2rem;
        box-shadow: 0 6px 22px rgba(40, 60, 120, .055); margin-bottom: .8rem;
    }
    .metric-card {
        background: white; border: 1px solid #e5eaff; border-radius: 18px;
        padding: 1rem 1.1rem; min-height: 120px;
        box-shadow: 0 5px 18px rgba(40, 60, 120, .045);
    }
    .metric-label { color: #64748b; font-size: .82rem; font-weight: 600; }
    .metric-value { color: #172554; font-family: 'Manrope', sans-serif; font-size: 1.65rem; font-weight: 800; margin: .2rem 0; }
    .metric-note { color: #64748b; font-size: .78rem; }
    .small-muted { color: #64748b; font-size: .9rem; }
    .result-high { background: #fff0f2; border: 1px solid #ffd5dc; border-radius: 18px; padding: 1rem 1.2rem; }
    .result-medium { background: #fff8e8; border: 1px solid #ffe7b0; border-radius: 18px; padding: 1rem 1.2rem; }
    .result-low { background: #eafbf4; border: 1px solid #c3f0dc; border-radius: 18px; padding: 1rem 1.2rem; }
    div.stButton > button[kind="primary"] {
        border: 0; border-radius: 12px; min-height: 3rem; font-weight: 700;
        background: linear-gradient(100deg, #18b7ad 0%, #5268e8 100%);
        color: white; box-shadow: 0 6px 16px rgba(73, 104, 220, .18);
    }
    div.stButton > button { border-radius: 11px; }
    div[data-testid="stSlider"] [data-baseweb="slider"] { padding-top: .2rem; }
    .footer-note { text-align:center; color:#8190ac; font-size:.78rem; padding:1rem 0 .3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Model loading --------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Run `python train_model.py` first, then make sure "
        "`stress_predictor_model.pkl` is in the same folder as `app.py`."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

# -------------------- Session state --------------------
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "history" not in st.session_state:
    st.session_state.history = []
if "completed_tasks" not in st.session_state:
    st.session_state.completed_tasks = []

# -------------------- Helpers --------------------
def stress_style(prediction):
    label = str(prediction).strip().lower()
    if "high" in label:
        return "high", "🔴", "High stress"
    if "medium" in label or "moderate" in label:
        return "medium", "🟠", "Moderate stress"
    return "low", "🟢", "Low stress"

def metric_card(label, value, note, emoji):
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-label">{emoji} &nbsp; {label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
        </div>""",
        unsafe_allow_html=True,
    )

def run_prediction(values):
    # Keep the feature order consistent with train_model.py:
    # age, study_hours, sleep_hours, academic_pressure, physical_activity
    row = np.array([[
        values["age"],
        values["study_hours"],
        values["sleep_hours"],
        values["academic_pressure"],
        values["physical_activity"],
    ]])
    return model.predict(row)[0]

def save_checkin(values, prediction):
    today = date.today().isoformat()
    entry = {
        "date": today,
        "prediction": str(prediction),
        "age": values["age"],
        "study_hours": values["study_hours"],
        "sleep_hours": values["sleep_hours"],
        "academic_pressure": values["academic_pressure"],
        "physical_activity": values["physical_activity"],
    }
    # Replace today's previous entry rather than duplicating it every click.
    st.session_state.history = [
        x for x in st.session_state.history if x["date"] != today
    ]
    st.session_state.history.append(entry)
    st.session_state.last_prediction = {"values": values.copy(), "prediction": str(prediction)}

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:.8rem .25rem 1.2rem">
          <div style="font-size:2.2rem">🎓</div>
          <div style="font-family:Manrope,sans-serif;font-size:1.35rem;font-weight:800">MindBalance AI</div>
          <div style="font-size:.83rem;opacity:.78">Student Stress & Wellness</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio(
        "NAVIGATION",
        ["🏠  Dashboard", "📝  Assessment", "🌿  Wellness Plan", "📈  Progress"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### 💙 Small steps matter")
    st.caption("A student wellness companion designed to encourage reflection and healthier routines.")
    st.markdown(
        '<div style="border:1px solid #42578a;border-radius:14px;padding:.8rem;margin-top:1.3rem;font-size:.82rem">🌍 <b>SDG 3</b><br/>Good Health and Well-being</div>',
        unsafe_allow_html=True,
    )

# -------------------- Header --------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Your personal wellness space</div>
      <h1>Understand your stress. Find your balance. ✨</h1>
      <p>Check in with your habits, explore your results, and build a routine that supports you.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Input form --------------------
def assessment_form(key_prefix="main"):
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("🧭 Daily wellness check-in")
    st.caption("Adjust the controls to match your current routine.")
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("🎂 Age", 15, 30, 20, key=f"{key_prefix}_age")
        study = st.slider("📚 Daily study hours", 0, 16, 6, key=f"{key_prefix}_study")
        sleep = st.slider("🌙 Daily sleep hours", 0, 12, 7, key=f"{key_prefix}_sleep")
    with c2:
        pressure = st.slider(
            "🧠 Academic pressure (1 = very low, 5 = high)",
            1, 5, 3, key=f"{key_prefix}_pressure"
        )
        activity = st.slider(
            "🏃 Exercise / physical activity (hours per week)",
            0, 14, 3, key=f"{key_prefix}_activity"
        )
        st.markdown(
            '<div style="background:#f0f5ff;border-radius:12px;padding:.8rem;margin-top:.3rem;color:#435578;font-size:.88rem">💡 <b>Tip:</b> Answer honestly. This is a reflection tool, not a test you can pass or fail.</div>',
            unsafe_allow_html=True,
        )
    values = {
        "age": age,
        "study_hours": study,
        "sleep_hours": sleep,
        "academic_pressure": pressure,
        "physical_activity": activity,
    }
    clicked = st.button("✨  Predict my stress level", type="primary", use_container_width=True, key=f"{key_prefix}_predict")
    st.markdown("</div>", unsafe_allow_html=True)
    return values, clicked

# -------------------- Dashboard page --------------------
if page == "🏠  Dashboard":
    st.markdown("### Welcome back 👋")
    st.markdown('<p class="small-muted">A quick look at your wellness today. Start with an assessment to refresh your insights.</p>', unsafe_allow_html=True)

    if st.session_state.last_prediction:
        last = st.session_state.last_prediction
        vals = last["values"]
        pred = last["prediction"]
        style, emoji, friendly = stress_style(pred)
        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Stress estimate", friendly, "Latest model prediction", emoji)
        with c2: metric_card("Sleep", f'{vals["sleep_hours"]} hrs', "Your reported sleep", "🌙")
        with c3: metric_card("Study time", f'{vals["study_hours"]} hrs', "Daily study hours", "📚")
        with c4: metric_card("Activity", f'{vals["physical_activity"]} hrs', "Exercise per week", "🏃")

        st.markdown("")
        if style == "high":
            st.markdown(f'<div class="result-high"><h3>🔴 {friendly}</h3><p>Consider a gentler pace today. Try a short break, a manageable study plan, and a consistent bedtime. If stress feels overwhelming or persistent, reach out to someone you trust or a qualified professional.</p></div>', unsafe_allow_html=True)
        elif style == "medium":
            st.markdown(f'<div class="result-medium"><h3>🟠 {friendly}</h3><p>You may benefit from balancing focused study sessions with rest, movement, and time away from screens.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-low"><h3>🟢 {friendly}</h3><p>Keep nurturing the habits that help you feel well. A low model estimate does not guarantee that everything is okay—your own feelings matter, too.</p></div>', unsafe_allow_html=True)

        st.markdown("### Your lifestyle snapshot")
        a, b, c = st.columns(3)
        with a:
            metric_card("Sleep insight", f'{vals["sleep_hours"]} hours', "Most adults need 7–9 hours; teens often need more.", "🌙")
        with b:
            metric_card("Study rhythm", f'{vals["study_hours"]} hours/day', "Try focused sessions with short breaks.", "📖")
        with c:
            metric_card("Movement", f'{vals["physical_activity"]} hrs/week', "Choose activity that feels accessible to you.", "🌿")
    else:
        c1, c2 = st.columns([1.25, 1])
        with c1:
            st.markdown('<div class="panel"><h3>🌈 Make space for yourself</h3><p>Small, consistent habits can support student well-being. Use this space to reflect on your routine, not to judge yourself.</p><ul><li>Notice how much rest you are getting.</li><li>Break large tasks into smaller steps.</li><li>Make time for movement and connection.</li></ul></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="panel"><h3>🚀 Get started</h3><p>Complete a quick check-in to receive a model-based stress category and practical wellness suggestions.</p></div>', unsafe_allow_html=True)
            if st.button("Go to assessment →", type="primary", use_container_width=True):
                st.session_state["nav_assessment"] = True
                st.info("Choose **Assessment** in the sidebar to continue.")

    st.markdown("### 🌟 A few gentle reminders")
    r1, r2, r3 = st.columns(3)
    with r1: metric_card("Take a pause", "5 minutes", "Step away, stretch, breathe, or get water.", "☕")
    with r2: metric_card("Protect your rest", "Consistent sleep", "A regular wind-down routine may help.", "🌙")
    with r3: metric_card("Ask for support", "You matter", "Talk with a trusted person when you need to.", "💙")

# -------------------- Assessment page --------------------
elif page == "📝  Assessment":
    st.markdown("### 📝 Student stress assessment")
    st.markdown('<p class="small-muted">Your answers are used to generate a prediction with your trained machine-learning model.</p>', unsafe_allow_html=True)
    values, clicked = assessment_form("assessment")
    if clicked:
        try:
            prediction = run_prediction(values)
            save_checkin(values, prediction)
            st.success("Your assessment is complete.")
            style, emoji, friendly = stress_style(prediction)
            if style == "high":
                st.markdown(f'<div class="result-high"><h2>🔴 Predicted Stress Level: {prediction}</h2><p>Consider prioritizing rest, manageable study goals, and supportive conversations.</p></div>', unsafe_allow_html=True)
            elif style == "medium":
                st.markdown(f'<div class="result-medium"><h2>🟠 Predicted Stress Level: {prediction}</h2><p>A balanced routine with study breaks and downtime may be helpful.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-low"><h2>🟢 Predicted Stress Level: {prediction}</h2><p>Keep checking in with yourself and maintaining supportive habits.</p></div>', unsafe_allow_html=True)

            st.markdown("#### Your inputs at a glance")
            x1, x2, x3 = st.columns(3)
            with x1: metric_card("Sleep", f'{values["sleep_hours"]} hrs', "Daily", "🌙")
            with x2: metric_card("Study", f'{values["study_hours"]} hrs', "Daily", "📚")
            with x3: metric_card("Activity", f'{values["physical_activity"]} hrs', "Per week", "🏃")

            if "high" in str(prediction).lower():
                st.info("If you are feeling distressed, your experience matters more than a model output. Consider talking to a trusted person, campus counsellor, or qualified health professional.")
            st.caption("This tool provides an estimate based on dataset patterns. It is not a medical diagnosis.")
        except Exception as exc:
            st.error(f"Prediction failed. Check that the input features and their order match your training script. Details: {exc}")

# -------------------- Wellness Plan page --------------------
elif page == "🌿  Wellness Plan":
    st.markdown("### 🌿 Your gentle wellness plan")
    st.markdown('<p class="small-muted">Pick small, realistic actions. You can tick off what you complete during this session.</p>', unsafe_allow_html=True)

    tasks = [
        ("☕", "Take a 5-minute reset", "Step away from your screen, stretch, or have a glass of water."),
        ("🌙", "Create a sleep wind-down", "Choose a realistic bedtime and reduce stimulating screen time before bed."),
        ("📚", "Try a focused study block", "Work on one task for 25 minutes, then take a short break."),
        ("🚶", "Add a little movement", "Take a short walk or do a few gentle stretches if comfortable."),
        ("💬", "Connect with someone", "Message a friend, family member, mentor, or trusted person."),
    ]
    for i, (icon, title, detail) in enumerate(tasks):
        with st.container(border=True):
            cols = st.columns([0.08, 0.68, 0.24])
            with cols[0]:
                st.markdown(f"<div style='font-size:1.55rem;padding-top:.35rem'>{icon}</div>", unsafe_allow_html=True)
            with cols[1]:
                st.markdown(f"**{title}**")
                st.caption(detail)
            with cols[2]:
                checked = st.checkbox("Done", key=f"wellness_task_{i}", value=i in st.session_state.completed_tasks)
                if checked and i not in st.session_state.completed_tasks:
                    st.session_state.completed_tasks.append(i)
                elif not checked and i in st.session_state.completed_tasks:
                    st.session_state.completed_tasks.remove(i)

    done = len(st.session_state.completed_tasks)
    st.progress(done / len(tasks), text=f"{done} of {len(tasks)} actions completed")
    if st.button("Reset checklist"):
        st.session_state.completed_tasks = []
        st.rerun()

    st.markdown('<div class="panel"><h3>💙 Remember</h3><p>Wellness is personal. Choose what works for you, and do not feel pressured to complete every item. If stress is persistent or difficult to manage, consider reaching out to a qualified professional or your campus support service.</p></div>', unsafe_allow_html=True)

# -------------------- Progress page --------------------
elif page == "📈  Progress":
    st.markdown("### 📈 Your check-in history")
    st.caption("Check-ins are kept in this app session only. They are not uploaded to a database.")
    history = st.session_state.history
    if not history:
        st.info("No saved check-ins yet. Complete an assessment to start your progress view.")
        if st.button("Go to assessment", type="primary"):
            st.info("Select **Assessment** in the sidebar.")
    else:
        df = pd.DataFrame(history)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        a, b, c = st.columns(3)
        with a: metric_card("Check-ins", len(df), "Saved this session", "🗓️")
        with b: metric_card("Latest prediction", df.iloc[-1]["prediction"], "Most recent check-in", "🧠")
        with c: metric_card("Avg. study time", f'{df["study_hours"].mean():.1f} hrs', "Across saved check-ins", "📚")

        # Categorical predictions are plotted as category labels, not as a numeric clinical score.
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["study_hours"], mode="lines+markers",
            name="Study hours", line=dict(color="#5268e8", width=3),
            marker=dict(size=8),
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["sleep_hours"], mode="lines+markers",
            name="Sleep hours", line=dict(color="#13a6a6", width=3),
            marker=dict(size=8),
        ))
        fig.update_layout(
            title="Sleep & study routine over your check-ins",
            height=370, margin=dict(l=10, r=10, t=55, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.12),
            xaxis=dict(showgrid=False, title="Date"),
            yaxis=dict(showgrid=True, gridcolor="#e8edfa", title="Hours"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### Recent check-ins")
        display = df[["date", "prediction", "sleep_hours", "study_hours", "academic_pressure", "physical_activity"]].copy()
        display["date"] = display["date"].dt.strftime("%d %b %Y")
        display = display.rename(columns={
            "date": "Date", "prediction": "Predicted category",
            "sleep_hours": "Sleep (h)", "study_hours": "Study (h)",
            "academic_pressure": "Academic pressure", "physical_activity": "Activity (h/week)"
        })
        st.dataframe(display, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download my check-ins (CSV)", csv, "mindbalance_checkins.csv", "text/csv")

# -------------------- Footer --------------------
st.markdown(
    '<div class="footer-note">MindBalance AI · Student wellness support · Model predictions are estimates, not diagnoses.</div>',
    unsafe_allow_html=True,
)
