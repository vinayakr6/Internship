# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from pathlib import Path

from utils import (
    load_data,
    preprocess_df,
    build_model,
    train_model,
    evaluate_model,
    predict_single_row,
    save_model,
    DEFAULT_NUMERIC_FEATURES,
)

st.set_page_config(page_title="Avocado ML App", layout="wide")
st.title("Avocado Price — Streamlit App (converted from notebook)")

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Sidebar: data source
st.sidebar.header("Data & Model")
data_source = st.sidebar.radio("Data source", ("Bundled CSV (data/avocado.csv)", "Upload CSV"))

df = None
if data_source.startswith("Bundled"):
    bundled_path = DATA_DIR / "avocado.csv"
    if bundled_path.exists():
        df = load_data(bundled_path)
    else:
        st.sidebar.error(f"No bundled CSV found at {bundled_path}. Please upload a CSV.")
        data_source = "Upload CSV"

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload avocado CSV", type=["csv"])
    if uploaded is not None:
        # read uploaded file into pandas
        df = load_data(io.BytesIO(uploaded.read()))

if df is None:
    st.info("Please upload a CSV or place a CSV at `data/avocado.csv` and refresh.")
    st.stop()

st.write(f"Dataset loaded — shape: {df.shape}")
if st.checkbox("Show raw data (first 100 rows)"):
    st.dataframe(df.head(100))

# Sidebar: pre-processing options & visualization filters
st.sidebar.header("EDA & Filters")
# let user pick columns for filters (commonly region, type, year)
filter_cols = []
for c in ["region", "type", "year", "date"]:
    if c in df.columns:
        filter_cols.append(c)

filters = {}
for c in filter_cols:
    vals = df[c].dropna().unique().tolist()
    chosen = st.sidebar.multiselect(f"Filter by {c}", options=sorted(vals), default=None)
    if chosen:
        filters[c] = chosen

# Apply filters
for c, chosen in filters.items():
    df = df[df[c].isin(chosen)]

st.sidebar.header("Model options")
model_action = st.sidebar.radio("Model action", ("Load pre-trained model", "Train model in-app (demo)"))

# Target & features selection
st.sidebar.header("Features & Target")
all_cols = df.columns.tolist()
default_target = "AveragePrice" if "AveragePrice" in all_cols else all_cols[-1]
target_col = st.sidebar.selectbox("Target column", options=all_cols, index=all_cols.index(default_target))
available_features = [c for c in all_cols if c != target_col]
feature_cols = st.sidebar.multiselect("Feature columns (select at least 1)", options=available_features, default=DEFAULT_NUMERIC_FEATURES if set(DEFAULT_NUMERIC_FEATURES).issubset(set(available_features)) else available_features[:6])

if len(feature_cols) == 0:
    st.sidebar.warning("Select at least one feature.")
    st.stop()

# Train/test split slider
test_size = st.sidebar.slider("Test set proportion", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

# Model hyperparameters (for RandomForestRegressor demo)
st.sidebar.subheader("RandomForest hyperparameters (demo)")
n_estimators = st.sidebar.slider("n_estimators", 10, 500, 100, step=10)
max_depth = st.sidebar.slider("max_depth (None=0)", 0, 100, 10, step=1)

# Model upload if loading
loaded_model = None
if model_action == "Load pre-trained model":
    model_file = st.sidebar.file_uploader("Upload model (.pkl / .joblib)", type=["pkl", "joblib"])
    if model_file is not None:
        try:
            loaded_model = joblib.load(model_file)
            st.sidebar.success("Pre-trained model loaded from upload.")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
    else:
        # Try load model/model.pkl in project
        local_path = MODEL_DIR / "model.pkl"
        if local_path.exists():
            if st.sidebar.button("Load model/model.pkl from project"):
                loaded_model = joblib.load(local_path)
                st.sidebar.success("Loaded model/model.pkl")

# Split & preprocess
if st.button("Run preprocessing & split"):
    with st.spinner("Preprocessing..."):
        X_train, X_test, y_train, y_test, preprocess_obj = preprocess_df(df, feature_cols, target_col, test_size=test_size)
    st.success(f"Preprocessing done — Train: {X_train.shape}, Test: {X_test.shape}")
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["preprocess"] = preprocess_obj

# If preprocessing already run in session, pull from session_state
X_train = st.session_state.get("X_train", None)
X_test = st.session_state.get("X_test", None)
y_train = st.session_state.get("y_train", None)
y_test = st.session_state.get("y_test", None)
preprocess_obj = st.session_state.get("preprocess", None)

# Allow training only after preprocessing
if model_action == "Train model in-app (demo)":
    if X_train is None:
        st.warning("Run preprocessing first (click the button) before training.")
    else:
        st.subheader("Training")
        if st.button("Train RandomForest (demo)"):
            with st.spinner("Training model..."):
                model = train_model(
                    "random_forest",
                    X_train,
                    y_train,
                    n_estimators=n_estimators,
                    max_depth=None if max_depth == 0 else max_depth,
                )
            st.success("Training complete.")
            st.session_state["model"] = model
            # Save model to model/model_demo.pkl
            demo_path = MODEL_DIR / "model_demo.pkl"
            save_model(model, demo_path)
            st.info(f"Saved demo model to {demo_path}")

# If loaded_model provided, put in session_state
if loaded_model is not None:
    st.session_state["model"] = loaded_model

# Show evaluation if model present
model = st.session_state.get("model", None)
if model is not None and X_test is not None:
    st.subheader("Evaluation")
    metrics, fig_actual_vs_pred, fig_residuals = evaluate_model(model, X_test, y_test, preprocess_obj=preprocess_obj)
    st.write("Metrics:", metrics)
    st.plotly_chart(fig_actual_vs_pred, use_container_width=True)
    st.plotly_chart(fig_residuals, use_container_width=True)
    if st.button("Save current model to model/model.pkl"):
        save_model(model, MODEL_DIR / "model.pkl")
        st.success(f"Model saved to {MODEL_DIR / 'model.pkl'}")

# EDA visualizations (Plotly)
st.subheader("Exploratory Data Analysis (Plotly)")
import plotly.express as px

eda_choice = st.selectbox("Choose visualization", ["Price over time", "Price by region", "Histogram of a feature", "Correlation heatmap"])
if eda_choice == "Price over time":
    if "date" in df.columns:
        df_plot = df.copy()
        # ensure date is datetime
        df_plot["date"] = pd.to_datetime(df_plot["date"], errors="coerce")
        df_plot = df_plot.sort_values("date")
        fig = px.line(df_plot, x="date", y=target_col, color="region" if "region" in df_plot.columns else None, title="Price over time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No 'date' column in dataset.")
elif eda_choice == "Price by region":
    if "region" in df.columns:
        fig = px.box(df, x="region", y=target_col, title="Price by region")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No 'region' column in dataset.")
elif eda_choice == "Histogram of a feature":
    feat = st.selectbox("Feature", options=feature_cols)
    fig = px.histogram(df, x=feat, nbins=30, title=f"Histogram of {feat}")
    st.plotly_chart(fig, use_container_width=True)
else:  # correlation heatmap
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation matrix")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough numeric columns for correlation heatmap.")

# Single-row prediction UI
st.subheader("Single prediction (manual input)")
if model is None:
    st.info("No model loaded or trained yet. Load a pre-trained model or train in-app to enable predictions.")
else:
    input_vals = {}
    st.write("Enter feature values (strings accepted — will attempt numeric conversion where applicable).")
    for f in feature_cols:
        default_val = ""
        if f in df.columns:
            default_val = str(df[f].dropna().iloc[0])
        input_vals[f] = st.text_input(f, value=default_val)
    if st.button("Predict"):
        row = {}
        for k, v in input_vals.items():
            # try convert to numeric, otherwise keep as string
            try:
                row[k] = float(v)
            except:
                row[k] = v
        pred = predict_single_row(model, row, preprocess_obj)
        st.success(f"Predicted {target_col}: {pred:.4f}")

st.markdown("---")
st.caption("App generated from Avocado Project notebook. Adjust code in utils.py to match any notebook-specific preprocessing or model choices.")
