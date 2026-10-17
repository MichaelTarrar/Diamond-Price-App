"""
Diamond Price Estimator - Assignment 3 web app

Loads diamond_price_model.pkl (the whole pipeline: pre-processing + best
algorithm, saved by the notebook) and predicts a diamond's price from six
characteristics. model_info.json supplies the category lists, the input
ranges and the cross-validation error.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(page_title="Diamond Price Estimator", page_icon="💎", layout="centered")

HERE = Path(__file__).parent
MODEL_PATH = HERE / "diamond_price_model.pkl"
INFO_PATH = HERE / "model_info.json"

# Cheapest diamond in the dataset. The model can drift below it for rare small,
# low-grade stones, so estimates are never shown under this value.
MIN_PRICE = 326.0


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

# Shown from the lowest grade to the highest
CUT_ORDER = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER = ["J", "I", "H", "G", "F", "E", "D"]
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


def ordered(preferred, available):
    return [o for o in preferred if o in available] or sorted(available)


cuts = ordered(CUT_ORDER, info["categories"]["cut"])
colors = ordered(COLOR_ORDER, info["categories"]["color"])
clarities = ordered(CLARITY_ORDER, info["categories"]["clarity"])
carat_lo, carat_hi = info["ranges"]["carat"]
depth_lo, depth_hi = info["ranges"]["depth"]
table_lo, table_hi = info["ranges"]["table"]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    html, body, .stApp, .stApp * { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
    .stApp { background: #ffffff; color: #111111; }
    .block-container { max-width: 620px; padding-top: 4.5rem; padding-bottom: 3rem; }
    #MainMenu, footer { visibility: hidden; }

    .app-title { font-size: 1.7rem; font-weight: 600; letter-spacing: -0.01em; margin: 0; color: #111111; }
    .app-sub { font-size: 0.95rem; color: #6b7280; margin: 0.4rem 0 2.2rem 0; }

    [data-testid="stWidgetLabel"] p { font-size: 0.82rem; font-weight: 500; color: #4b5563; }
    div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background: #ffffff; border: 1px solid #e5e7eb; border-radius: 6px; box-shadow: none;
    }
    div[data-baseweb="select"] div, input { color: #111111 !important; }

    .stButton > button {
        background: #111111; color: #ffffff; border: none; border-radius: 6px;
        font-weight: 500; padding: 0.6rem 1rem; box-shadow: none;
    }
    .stButton > button:hover { background: #374151; color: #ffffff; }
    .stButton > button:focus { box-shadow: none; color: #ffffff; }

    .result { border-top: 1px solid #e5e7eb; margin-top: 2.2rem; padding-top: 1.6rem; }
    .result-label { font-size: 0.82rem; color: #6b7280; margin: 0; }
    .result-price { font-size: 3rem; font-weight: 600; letter-spacing: -0.02em; margin: 0.1rem 0 0.2rem 0; color: #111111; }
    .result-range { font-size: 0.95rem; color: #4b5563; margin: 0; }
    .result-note { font-size: 0.8rem; color: #9ca3af; margin: 0.9rem 0 0 0; }

    [data-testid="stCaptionContainer"] { color: #9ca3af; font-size: 0.78rem; }
    </style>

    <p class="app-title">Diamond Price Estimator</p>
    <p class="app-sub">Enter the diamond's details to get an estimated price in US dollars.</p>
    """,
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    carat = st.number_input("Carat", min_value=float(carat_lo), max_value=float(carat_hi),
                            value=1.00, step=0.01, format="%.2f")
    color = st.selectbox("Colour (J lowest, D highest)", colors,
                         index=colors.index("G") if "G" in colors else 0)
    depth = st.number_input("Depth (%)", min_value=float(depth_lo), max_value=float(depth_hi),
                            value=61.8, step=0.1, format="%.1f")
with col2:
    cut = st.selectbox("Cut", cuts, index=len(cuts) - 1)
    clarity = st.selectbox("Clarity (I1 lowest, IF highest)", clarities,
                           index=clarities.index("VS2") if "VS2" in clarities else 0)
    table = st.number_input("Table (%)", min_value=float(table_lo), max_value=float(table_hi),
                            value=57.0, step=0.1, format="%.1f")

if st.button("Estimate price", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "carat": carat, "depth": depth, "table": table,
        "cut": cut, "color": color, "clarity": clarity,
    }])[info["numeric_features"] + info["categorical_features"]]

    price = max(float(model.predict(row)[0]), MIN_PRICE)
    mae = info["mean_mae"]
    low, high = max(price - mae, MIN_PRICE), price + mae

    st.markdown(
        f"""
        <div class="result">
            <p class="result-label">Estimated price</p>
            <p class="result-price">${price:,.0f}</p>
            <p class="result-range">Typical range ${low:,.0f} to ${high:,.0f}</p>
            <p class="result-note">
                Range = the model's average absolute error (${mae:,.0f}), measured with
                {info['cv_folds']}-fold cross validation. Expensive diamonds can be off by more.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if carat > 2.5:
        st.warning(
            "Very large diamonds are rare in the training data (about 0.2% are above 2.5 carat), "
            "so this estimate is probably too low."
        )

    with st.expander("Input sent to the model"):
        st.dataframe(row, hide_index=True)

st.write("")
st.caption(
    f"{info['algorithm']} · cross-validated RMSE ${info['mean_rmse']:,.0f} · "
    "AIC354 Assignment 3 teaching demo, not a jeweller's appraisal."
)
if info.get("sklearn_version") != sklearn.__version__:
    st.caption(
        f"Model saved with scikit-learn {info.get('sklearn_version')}, app runs {sklearn.__version__}. "
        "If predictions fail, make requirements.txt use the first version."
    )
