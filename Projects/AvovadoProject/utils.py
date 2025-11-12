# utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Any

# sensible default numeric features (will be used as default selection if present)
DEFAULT_NUMERIC_FEATURES = ["Total Volume", "4046", "4225", "4770", "Total Bags", "Small Bags", "Large Bags", "XLarge Bags"]

def load_data(source) -> pd.DataFrame:
    """
    Load CSV from path-like or file-like object and return DataFrame.
    """
    if isinstance(source, (str, bytes, bytearray)):
        df = pd.read_csv(source)
    else:
        # file-like (BytesIO)
        df = pd.read_csv(source)
    return df

def preprocess_df(df: pd.DataFrame, feature_cols, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Splits and creates a preprocessing pipeline:
      - numeric: StandardScaler
      - categorical: OneHotEncoder (drop='first')
    Returns X_train, X_test, y_train, y_test, preprocess_pipeline
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # detect numeric and categorical features
    numeric_feats = X.select_dtypes(include=np.number).columns.tolist()
    categorical_feats = [c for c in feature_cols if c not in numeric_feats]

    num_pipeline = Pipeline([("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False, drop='first'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_feats),
            ("cat", cat_pipeline, categorical_feats),
        ],
        remainder="drop",
    )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # fit preprocessor on training data
    preprocessor.fit(X_train_raw)

    # transform
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    return X_train, X_test, y_train.values, y_test.values, (preprocessor, numeric_feats, categorical_feats)

def build_model(model_name: str = "random_forest", **kwargs):
    if model_name == "random_forest":
        n_estimators = kwargs.get("n_estimators", 100)
        max_depth = kwargs.get("max_depth", None)
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    elif model_name == "linear_regression":
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return model

def train_model(model_name: str, X_train, y_train, **kwargs):
    model = build_model(model_name, **kwargs)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, preprocess_obj=None):
    """
    Returns metrics dict and two plotly figures: actual_vs_pred and residuals.
    If preprocess_obj is provided (preprocessor, numeric_feats, categorical_feats), it is used for axis labels where possible.
    """
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    metrics = {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

    # actual vs predicted
    fig1 = px.scatter(x=y_test, y=preds, labels={"x": "Actual", "y": "Predicted"}, title="Actual vs Predicted")
    fig1.add_shape(type="line", x0=min(y_test.min(), preds.min()), x1=max(y_test.max(), preds.max()), y0=min(y_test.min(), preds.min()), y1=max(y_test.max(), preds.max()), line=dict(color="red", dash="dash"))

    # residuals
    residuals = y_test - preds
    fig2 = px.scatter(x=preds, y=residuals, labels={"x": "Predicted", "y": "Residual"}, title="Residuals vs Predicted")
    fig2.add_hline(y=0, line_dash="dash", line_color="red")

    return metrics, fig1, fig2

def predict_single_row(model, row: dict, preprocess_obj: tuple):
    """
    row: dict of feature_name -> value (single row).
    preprocess_obj: (preprocessor, numeric_feats, categorical_feats)
    """
    preprocessor, numeric_feats, categorical_feats = preprocess_obj
    # create df for a single row
    # ensure columns order
    cols = numeric_feats + categorical_feats
    row_df = pd.DataFrame([row], columns=cols)
    X_proc = preprocessor.transform(row_df)
    pred = model.predict(X_proc)
    return float(pred[0])

def save_model(model: Any, path):
    joblib.dump(model, path)
