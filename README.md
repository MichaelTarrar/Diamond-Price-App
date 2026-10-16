# Diamond Price Predictor

AIC354 Machine Learning Fundamentals, Assignment 3 (Task 2).
Student: Michael Hassan, FA24-BSE-064, Section A.

A Streamlit web app that predicts the price of a diamond from its carat, cut,
colour, clarity, depth and table. The model is the best of five algorithms
compared with 10-fold cross validation on the full diamonds dataset
(53,940 rows before cleaning).

## Files

| File | Purpose |
|---|---|
| `app.py` | The web app |
| `diamond_price_model.pkl` | Trained pipeline (pre-processing + best algorithm) |
| `model_info.json` | Algorithm name, cross-validation error, categories and input ranges |
| `requirements.txt` | Libraries Streamlit Cloud installs |

## Run it on your own computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this folder to a **public** GitHub repository (files at the top level, not inside a sub-folder).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Create app -> Deploy a public app from GitHub.
4. Repository: your repo, Branch: `main`, Main file path: `app.py`.
5. Deploy, then copy the `.streamlit.app` link into the top of the notebook.

## Important: scikit-learn version

A `.pkl` file must be loaded with the same scikit-learn version that saved it.
The version is stored in `model_info.json` (`sklearn_version`) and `requirements.txt`
must match it. If you re-train the model in Google Colab, replace the `.pkl` and
`.json` files **and** change the version line in `requirements.txt`.
