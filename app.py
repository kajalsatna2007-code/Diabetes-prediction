"""Diabetes Risk Check - Streamlit app for the logistic regression notebook.

Run:  streamlit run app.py
Needs model.pkl and scaler.pkl in the same folder as this file.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Check", page_icon="🩺", layout="wide")

BASE_DIR = Path(__file__).parent

# Column order must match the notebook's X (df.drop('diabetes', axis=1)).
FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
]

# LabelEncoder sorts classes alphabetically, so these codes match the notebook:
# gender: Female=0, Male=1, Other=2
# smoking_history: No Info=0, current=1, ever=2, former=3, never=4, not current=5
GENDERS = ["Female", "Male", "Other"]
SMOKING_LABELS = {
    "No info": "No Info",
    "Current smoker": "current",
    "Ever smoked": "ever",
    "Former smoker": "former",
    "Never smoked": "never",
    "Not current": "not current",
}
SMOKING_CLASSES = sorted(SMOKING_LABELS.values())

INK = "#12263A"
PAPER = "#F3F6F8"
TEAL = "#0E7C7B"
AMBER = "#C97A12"
RED = "#B8372B"
MUTED = "#5B6B7A"

# ----------------------------------------------------------------- styling
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;800&display=swap');

    html, body, .stApp, [class*="css"] {{ font-family: 'Manrope', system-ui, sans-serif; }}
    .stApp {{ background: {PAPER}; color: {INK}; }}
    .block-container {{ max-width: 1150px; padding-top: 2.2rem; }}
    h1, h2, h3, h4, label, p, span, li {{ color: {INK}; }}

    .title {{ font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em; margin: 0; }}
    .subtitle {{ color: {MUTED}; font-size: 1.02rem; margin: .25rem 0 1.4rem; max-width: 60ch; }}

    .panel {{ background: #fff; border: 1px solid #DCE4EA; border-radius: 10px; padding: 1.3rem 1.4rem; }}
    .result-label {{ color: {MUTED}; font-size: .9rem; margin-bottom: .1rem; }}
    .result-value {{ font-size: 3.2rem; font-weight: 800; line-height: 1.05; letter-spacing: -0.03em; }}
    .verdict {{ font-size: 1.15rem; font-weight: 600; margin: .2rem 0 1rem; }}

    .meter {{ position: relative; height: 12px; background: #E4EBF0; border-radius: 6px; overflow: hidden; }}
    .meter-fill {{ height: 100%; border-radius: 6px; transition: width .4s ease; }}
    .meter-scale {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: .78rem; margin-top: .35rem; }}

    .reading {{ display: flex; align-items: center; justify-content: space-between;
               padding: .65rem 0; border-bottom: 1px solid #E8EEF2; }}
    .reading:last-child {{ border-bottom: none; }}
    .reading .name {{ font-weight: 600; }}
    .reading .val {{ color: {MUTED}; font-size: .92rem; }}
    .pill {{ font-size: .8rem; font-weight: 600; padding: .15rem .65rem; border-radius: 999px; }}
    .pill.ok {{ background: #DDF1EF; color: #0A5F5E; }}
    .pill.warn {{ background: #FBEBD3; color: #8A5209; }}
    .pill.bad {{ background: #F7DEDA; color: #8F2A20; }}

    .note {{ color: {MUTED}; font-size: .85rem; margin-top: 1rem; }}
    div[data-testid="stTabs"] button {{ font-weight: 600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------- training + pkl creation
MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
CSV_NAME = "diabetes_prediction_dataset.csv"
KAGGLE_DATASET = "iammustafatz/diabetes-prediction-dataset"


def get_training_data():
    """Local CSV next to app.py first, then Kaggle. Returns None if neither works."""
    local_csv = BASE_DIR / CSV_NAME
    if local_csv.exists():
        return pd.read_csv(local_csv)
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        return kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, KAGGLE_DATASET, CSV_NAME)
    except Exception:
        return None


def train_and_save(df):
    """Same steps as the notebook: encode -> 80/20 split -> scale -> train -> save pkl files."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    missing = [c for c in FEATURES + ["diabetes"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")

    df = df.copy()
    df["gender"] = df["gender"].map(GENDERS.index)
    df["smoking_history"] = df["smoking_history"].map(SMOKING_CLASSES.index)

    X = df[FEATURES]
    y = df["diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 score": f1_score(y_test, y_pred),
    }


def ensure_artifacts():
    """If model.pkl / scaler.pkl are missing, offer to create them from the dataset."""
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        return

    st.markdown('<p class="title">Diabetes risk check</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">First-time setup: the app will train the logistic regression model '
        "and create <b>model.pkl</b> and <b>scaler.pkl</b> in this folder.</p>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        f"Dataset CSV (optional - skip it to use a local {CSV_NAME} or download from Kaggle)",
        type="csv",
    )
    if st.button("Train model and create pkl files", type="primary"):
        try:
            with st.spinner("Loading data and training the model..."):
                df = pd.read_csv(uploaded) if uploaded is not None else get_training_data()
                if df is None:
                    st.error(
                        "Could not load the dataset automatically. Upload "
                        f"{CSV_NAME} above or place it next to app.py, then try again."
                    )
                    st.stop()
                st.session_state["metrics"] = train_and_save(df)
        except ValueError as err:
            st.error(str(err))
            st.stop()
        st.rerun()
    st.stop()


ensure_artifacts()


@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


model, scaler = load_artifacts()


# ----------------------------------------------------------------- helpers
def band(value, cutoffs, labels):
    """Return label for the first cutoff the value is below, else the last label."""
    for cutoff, label in zip(cutoffs, labels):
        if value < cutoff:
            return label
    return labels[-1]


def reading_row(name, value_text, status):
    css = {"Normal": "ok", "Elevated": "warn", "High": "bad", "Yes": "bad", "No": "ok"}.get(status, "warn")
    return (
        f'<div class="reading"><div><div class="name">{name}</div>'
        f'<div class="val">{value_text}</div></div>'
        f'<span class="pill {css}">{status}</span></div>'
    )


# ------------------------------------------------------------------ header
st.markdown('<p class="title">Diabetes risk check</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Enter a patient\'s health details and the logistic regression '
    "model estimates the probability of diabetes. The result updates as you change the values.</p>",
    unsafe_allow_html=True,
)

tab_check, tab_drivers, tab_about = st.tabs(["Risk check", "What drives the model", "About"])

# --------------------------------------------------------------- tab: check
with tab_check:
    left, right = st.columns([5, 6], gap="large")

    with left:
        st.markdown("#### Patient details")
        c1, c2 = st.columns(2)
        gender = c1.selectbox("Gender", GENDERS, index=1)
        age = c2.number_input("Age (years)", min_value=1, max_value=100, value=50, step=1)

        smoking_label = st.selectbox("Smoking history", list(SMOKING_LABELS), index=4)

        c3, c4 = st.columns(2)
        hypertension = c3.toggle("Hypertension")
        heart_disease = c4.toggle("Heart disease")

        bmi = st.slider("BMI", min_value=10.0, max_value=70.0, value=28.0, step=0.1)
        hba1c = st.slider("HbA1c level (%)", min_value=3.5, max_value=9.0, value=6.0, step=0.1)
        glucose = st.slider("Blood glucose level (mg/dL)", min_value=80, max_value=300, value=140, step=1)

    # ---- prediction (same preprocessing as the notebook: encode -> scale -> predict)
    patient = pd.DataFrame(
        [
            {
                "gender": GENDERS.index(gender),
                "age": age,
                "hypertension": int(hypertension),
                "heart_disease": int(heart_disease),
                "smoking_history": SMOKING_CLASSES.index(SMOKING_LABELS[smoking_label]),
                "bmi": bmi,
                "HbA1c_level": hba1c,
                "blood_glucose_level": glucose,
            }
        ]
    )[FEATURES]

    scaled = scaler.transform(patient)
    proba = float(model.predict_proba(scaled)[0][1])
    predicted = int(model.predict(scaled)[0])

    if proba < 0.35:
        color, verdict = TEAL, "Low risk"
    elif proba < 0.50:
        color, verdict = AMBER, "Borderline"
    else:
        color, verdict = RED, "High risk - model predicts diabetes"

    with right:
        st.markdown("#### Result")
        st.markdown(
            f"""
            <div class="panel">
              <div class="result-label">Estimated probability of diabetes</div>
              <div class="result-value" style="color:{color}">{proba * 100:.1f}%</div>
              <div class="verdict" style="color:{color}">{verdict}</div>
              <div class="meter"><div class="meter-fill" style="width:{proba * 100:.1f}%; background:{color}"></div></div>
              <div class="meter-scale"><span>0%</span><span>Decision threshold 50%</span><span>100%</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown("##### How the inputs compare with common reference ranges")

        hba1c_status = band(hba1c, [5.7, 6.5], ["Normal", "Elevated", "High"])
        glucose_status = band(glucose, [140, 200], ["Normal", "Elevated", "High"])
        bmi_status = band(bmi, [25, 30], ["Normal", "Elevated", "High"])
        bmi_text = f"{bmi:.1f} " + band(bmi, [18.5, 25, 30], ["(underweight)", "(healthy)", "(overweight)", "(obese)"])

        rows = "".join(
            [
                reading_row("HbA1c", f"{hba1c:.1f}%", hba1c_status),
                reading_row("Blood glucose", f"{glucose} mg/dL", glucose_status),
                reading_row("BMI", bmi_text, bmi_status),
                reading_row("Hypertension", "Diagnosed" if hypertension else "Not reported", "Yes" if hypertension else "No"),
                reading_row("Heart disease", "Diagnosed" if heart_disease else "Not reported", "Yes" if heart_disease else "No"),
            ]
        )
        st.markdown(f'<div class="panel">{rows}</div>', unsafe_allow_html=True)

        st.markdown(
            '<p class="note">This tool is a learning project and not a medical diagnosis. '
            "Please consult a doctor for any health decision.</p>",
            unsafe_allow_html=True,
        )

# ------------------------------------------------------------- tab: drivers
with tab_drivers:
    st.markdown("#### Which features push the prediction up or down")
    if hasattr(model, "coef_"):
        coefs = pd.Series(model.coef_[0], index=FEATURES).sort_values()
        st.bar_chart(coefs, horizontal=True, color=TEAL, height=360)
        st.caption(
            "Coefficients of the logistic regression on standardized features. "
            "Positive values raise the predicted diabetes probability, negative values lower it, "
            "and a longer bar means a stronger effect."
        )
        top = coefs.abs().sort_values(ascending=False).index[:2]
        st.write(f"The two strongest features in this model are **{top[0]}** and **{top[1]}**.")
    else:
        st.info("This model does not expose coefficients.")

# --------------------------------------------------------------- tab: about
with tab_about:
    st.markdown("#### About this model")
    st.write(
        "The model is a **logistic regression** (with balanced class weights) trained on the "
        "Kaggle *Diabetes Prediction Dataset* of 100,000 records. It looks at eight inputs: "
        "gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level and blood glucose level."
    )
    st.write(
        "Inputs are label-encoded and standardized exactly as in the notebook before the model "
        "predicts. Because the classes are balanced during training, the probability is best read as a "
        "screening signal, not as a calibrated clinical risk."
    )

    metrics = st.session_state.get("metrics")
    if metrics:
        st.markdown("##### Test-set performance (from the training you just ran)")
        for col, (name, value) in zip(st.columns(len(metrics)), metrics.items()):
            col.metric(name, f"{value:.2f}")
