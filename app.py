"""
Diamond Price Predictor - Assignment 3 web app
----------------------------------------------------------
Loads the trained pipeline (diamond_price_model.pkl) produced by the notebook
"diamond_price_kfold_full_dataset_10fold.ipynb" and predicts the price of a
diamond from six characteristics.

The pickle contains the WHOLE pipeline (imputer + scaler + one-hot encoder +
algorithm), so raw values can be passed straight to model.predict().
model_info.json (also written by the notebook) supplies the category lists,
the valid input ranges and the cross-validation error shown under the result.

Visual identity: a grading-certificate look (ice-blue paper, ink text, teal
accents, double border), deliberately different from the other course apps.
"""

import json
from pathlib import Path

import pandas as pd
import pickle
import sklearn
import streamlit as st

st.set_page_config(page_title="Diamond Price Certificate", page_icon="💎", layout="centered")

HERE = Path(__file__).parent
MODEL_PATH = HERE / "diamond_price_model.pkl"
INFO_PATH = HERE / "model_info.json"


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_info():
    with open(INFO_PATH) as f:
        return json.load(f)


model = load_model()
info = load_info()

# Cut, colour and clarity are listed from worst to best, which is easier to read
# than the alphabetical order stored in model_info.json.
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


def ordered(preferred, available):
    """Keep only the options the model knows, in the preferred order."""
    return [o for o in preferred if o in available] or sorted(available)


cuts = ordered(CUT_ORDER, info["categories"]["cut"])
colors = ordered(COLOR_ORDER, info["categories"]["color"])
clarities = ordered(CLARITY_ORDER, info["categories"]["clarity"])
carat_lo, carat_hi = info["ranges"]["carat"]
depth_lo, depth_hi = info["ranges"]["depth"]
table_lo, table_hi = info["ranges"]["table"]

# ---------------------------------------------------------------------------
# Visual identity
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Work+Sans:wght@400;500;600&display=swap');

    :root {
        --paper: #EEF3F7;
        --ink: #14202B;
        --ink-soft: #4B5B69;
        --teal: #1F8A9E;
        --teal-deep: #176B7B;
        --line: #B9C7D3;
        --card: #FBFDFE;
    }

    .stApp { background: var(--paper); color: var(--ink); font-family: 'Work Sans', sans-serif; }
    .block-container { padding-top: 2.2rem; max-width: 760px; }

    .cert-frame {
        border: 1px solid var(--ink);
        outline: 1px solid var(--line);
        outline-offset: 5px;
        background: var(--card);
        padding: 1.8rem 2rem 1.5rem 2rem;
        margin: 0 0 2rem 0;
        text-align: center;
    }
    .cert-kicker { color: var(--teal-deep); font-size: 0.82rem; letter-spacing: 0.12em; margin: 0 0 0.5rem 0; }
    .cert-title { font-family: 'Fraunces', serif; font-weight: 700; font-size: 2.3rem; line-height: 1.1; margin: 0; }
    .cert-sub { color: var(--ink-soft); font-size: 0.95rem; margin: 0.8rem auto 0 auto; max-width: 34rem; }

    .section-title { font-family: 'Fraunces', serif; font-weight: 500; font-size: 1.25rem; margin: 0.3rem 0 0.2rem 0; }
    .section-hint { color: var(--ink-soft); font-size: 0.88rem; margin: 0 0 0.9rem 0; }

    label, [data-testid="stWidgetLabel"] p { color: var(--ink) !important; font-weight: 500 !important; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background: var(--card) !important; border: 1px solid var(--line) !important; border-radius: 4px;
    }

    .stButton > button {
        background: var(--teal); color: #fff; border: none; border-radius: 4px;
        font-family: 'Work Sans', sans-serif; font-weight: 600; padding: 0.65rem 1rem;
    }
    .stButton > button:hover { background: var(--teal-deep); color: #fff; }

    .result-card {
        border: 1px solid var(--ink); outline: 1px solid var(--line); outline-offset: 5px;
        background: var(--card); padding: 1.6rem 2rem; margin: 2rem 0 1rem 0; text-align: center;
    }
    .result-label { color: var(--teal-deep); font-size: 0.82rem; letter-spacing: 0.12em; margin: 0; }
    .result-price { font-family: 'Fraunces', serif; font-weight: 700; font-size: 3.4rem; margin: 0.2rem 0 0.3rem 0; }
    .result-range { color: var(--ink-soft); font-size: 0.98rem; margin: 0; }
    .result-note { color: var(--ink-soft); font-size: 0.85rem; margin: 1rem 0 0 0; border-top: 1px solid var(--line); padding-top: 0.8rem; }

    .stCaption, [data-testid="stCaptionContainer"] { color: var(--ink-soft) !important; }
    </style>

    <div class="cert-frame">
        <p class="cert-kicker">PRICE ESTIMATE · MACHINE LEARNING</p>
        <p class="cert-title">Diamond Price Certificate</p>
        <p class="cert-sub">
            Describe a diamond by its carat, cut, colour, clarity, depth and table.
            A trained model returns an estimated price in US dollars.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
st.markdown('<p class="section-title">The diamond</p>', unsafe_allow_html=True)
st.markdown('<p class="section-hint">Cut, colour and clarity are listed from the lowest grade to the highest.</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    carat = st.number_input("Carat (weight)", min_value=float(carat_lo), max_value=float(carat_hi),
                            value=1.00, step=0.01, format="%.2f")
    cut = st.selectbox("Cut", cuts, index=len(cuts) - 1)
with col2:
    color = st.selectbox("Colour (J lowest, D highest)", colors, index=colors.index("G") if "G" in colors else 0)
    clarity = st.selectbox("Clarity (I1 lowest, IF highest)", clarities,
                           index=clarities.index("VS2") if "VS2" in clarities else 0)

col3, col4 = st.columns(2)
with col3:
    depth = st.slider("Depth (%)", float(depth_lo), float(depth_hi), 61.8, 0.1)
with col4:
    table = st.slider("Table (%)", float(table_lo), float(table_hi), 57.0, 0.1)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if st.button("Estimate price", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "carat": carat, "depth": depth, "table": table,
        "cut": cut, "color": color, "clarity": clarity,
    }])[info["numeric_features"] + info["categorical_features"]]

    price = float(model.predict(row)[0])
    price = max(price, 0.0)
    mae = info["mean_mae"]
    low, high = max(price - mae, 0.0), price + mae

    st.markdown(
        f"""
        <div class="result-card">
            <p class="result-label">ESTIMATED PRICE</p>
            <p class="result-price">${price:,.0f}</p>
            <p class="result-range">Typical range: ${low:,.0f} to ${high:,.0f}</p>
            <p class="result-note">
                The range is the model's average absolute error (${mae:,.0f}) measured with
                {info['cv_folds']}-fold cross validation on {info['training_rows']:,} diamonds.
                Expensive diamonds can be off by more.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if carat > 2.5:
        st.warning(
            "Very large diamonds are rare in the training data (about 0.2% are above 2.5 carat), "
            "and this kind of model cannot predict far above the prices it has seen. "
            "The estimate is probably too low."
        )

    with st.expander("See the exact input sent to the model"):
        st.dataframe(row, hide_index=True)

st.divider()
st.caption(
    f"Model: {info['algorithm']} · cross-validated RMSE ${info['mean_rmse']:,.0f} (± ${info['std_rmse']:,.0f}) · "
    "AIC354 Assignment 3 teaching demo. Estimates reflect one public dataset and are not a jeweller's appraisal."
)
if info.get("sklearn_version") != sklearn.__version__:
    st.caption(
        f"⚠ Model was saved with scikit-learn {info.get('sklearn_version')}, this app runs {sklearn.__version__}. "
        "If predictions fail, make requirements.txt use the first version."
    )
