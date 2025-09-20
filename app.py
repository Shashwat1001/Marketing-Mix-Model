#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit MMM App — models + errors + flexible graphing (corr/box/scatter/contribution/categorical)
Run: streamlit run app.py
"""
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from sklearn.neural_network import MLPRegressor
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Marketing Mix Model (MMM) — Models + Graphs", layout="wide")

st.title("Marketing Mix Model — Evaluation + Graphs")

with st.sidebar:
    st.header("Upload & Setup")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    test_size = st.slider("Test size", 0.05, 0.5, 0.2, 0.05, help="Fraction for test split")
    random_state = st.number_input("Random state", value=42, step=1)

    st.markdown("---")
    st.subheader("Select Models")
    model_choices = {
    "Decision Tree": DecisionTreeRegressor(random_state=random_state),
    "KNN Regressor": neighbors.KNeighborsRegressor(n_neighbors=5),
    "MLP Regressor": MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=random_state),
    "AdaBoost": AdaBoostRegressor(random_state=random_state, n_estimators=200),
    "Gradient Boosting": GradientBoostingRegressor(random_state=random_state, n_estimators=200),
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state) if 'random_state' in Ridge().get_params() else Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=10000, tol=0.001),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1
        )
    }
    if HAS_XGB:
        model_choices["XGBoost"] = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=random_state, n_jobs=-1
        )

    chosen_models = st.multiselect("Choose algorithms", list(model_choices.keys()),
                                   default=["Linear Regression", "Random Forest"])

    st.markdown("---")
    st.subheader("Target & Features")
    auto_numeric_only = st.checkbox("Use only numeric features (faster)", value=False)

@st.cache_data(show_spinner=False)
def read_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def safe_infer_types(df: pd.DataFrame):
    dtypes = df.dtypes
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols

def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.where(y_true==0, np.nan, y_true), 1e-9, None))) * 100.0)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "MAE": mae, "MAPE (%)": mape, "R²": r2}

def build_preprocessor(df, target, numeric_only=False):
    num_cols, cat_cols = safe_infer_types(df.drop(columns=[target], errors="ignore"))
    if numeric_only:
        num_cols = [c for c in num_cols if c != target]
        cat_cols = []
    else:
        num_cols = [c for c in num_cols if c != target]
    # OHE for categoricals; passthrough numeric
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return pre

def get_feature_names(preprocessor, input_df, target):
    # Build names from the preprocessor after fitting
    num_cols, cat_cols = safe_infer_types(input_df.drop(columns=[target], errors="ignore"))
    feature_names = []
    try:
        # After fit, we can access transformers_
        for name, trans, cols in preprocessor.transformers_:
            if name == "num" and trans == "passthrough":
                feature_names.extend(cols)
            elif name == "cat":
                # OneHotEncoder
                ohe: OneHotEncoder = trans
                cats = ohe.get_feature_names_out(cols)
                feature_names.extend(list(cats))
    except Exception:
        # Fallback before fit
        feature_names = [c for c in input_df.columns if c != target]
    return feature_names

def train_and_eval(df: pd.DataFrame, target: str, models: dict, test_size: float, random_state: int, numeric_only: bool):
    X = df.drop(columns=[target])
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    pre = build_preprocessor(df, target, numeric_only=numeric_only)

    results = []
    fitted = {}

    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        m = metrics(y_test, y_pred)
        results.append({"Model": name, **m})
        fitted[name] = pipe

    res_df = pd.DataFrame(results).sort_values(by="MSE").reset_index(drop=True)
    return res_df, fitted, (X_train, X_test, y_train, y_test)

def draw_corr(df: pd.DataFrame):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        st.info("Need at least 2 numeric columns for correlation heatmap.")
        return
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", square=False)
    st.pyplot(fig)

def draw_box(df: pd.DataFrame, cols: list, by: str | None):
    if not cols:
        st.info("Choose at least one numeric column for box plot.")
        return
    for col in cols:
        if by and by in df.columns:
            fig, ax = plt.subplots(figsize=(7,5))
            sns.boxplot(data=df, x=by, y=col, ax=ax)
            ax.set_title(f"Box plot of {col} by {by}")
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(7,5))
            sns.boxplot(data=df[[col]], ax=ax)
            ax.set_title(f"Box plot of {col}")
            st.pyplot(fig)

def draw_scatter(df: pd.DataFrame, x: str, y: str, hue: str | None):
    if x is None or y is None:
        st.info("Pick X and Y columns for scatter.")
        return
    fig, ax = plt.subplots(figsize=(7,5))
    if hue and hue in df.columns and not pd.api.types.is_numeric_dtype(df[hue]):
        sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
    else:
        sns.scatterplot(data=df, x=x, y=y, ax=ax)
    ax.set_title(f"Scatter: {x} vs {y}")
    st.pyplot(fig)

def draw_categorical_bar(df: pd.DataFrame, cat_col: str, agg_on: str | None, agg_fn: str = "count"):
    if cat_col is None or cat_col not in df.columns:
        st.info("Pick a categorical column.")
        return
    if agg_fn == "count":
        ser = df[cat_col].value_counts().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x=ser.values, y=ser.index, ax=ax)
        ax.set_title(f"Count by {cat_col}")
        st.pyplot(fig)
    else:
        if agg_on is None or agg_on not in df.columns:
            st.info("Pick a numeric column to aggregate.")
            return
        grouped = df.groupby(cat_col)[agg_on].agg(agg_fn).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x=grouped.values, y=grouped.index, ax=ax)
        ax.set_title(f"{agg_fn.title()} of {agg_on} by {cat_col}")
        st.pyplot(fig)

def contribution_from_linear(pipe: Pipeline, X: pd.DataFrame, feature_names: list[str], use_mean: bool = True):
    # For linear models only; uses coefficients * values for a single row (or mean row)
    model = pipe.named_steps["model"]
    pre: ColumnTransformer = pipe.named_steps["pre"]
    if not hasattr(model, "coef_"):
        return None, None

    # Build a representative row
    row = X.mean(numeric_only=True).to_dict()
    # For categoricals, take mode
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            try:
                row[c] = X[c].mode(dropna=True).iloc[0]
            except Exception:
                row[c] = X[c].dropna().iloc[0] if len(X[c].dropna()) else None
    row_df = pd.DataFrame([row])

    # Transform
    Z = pre.transform(row_df)  # 2D
    coefs = getattr(model, "coef_", None).ravel()
    if coefs is None or Z.shape[1] != len(coefs):
        return None, None

    contrib = coefs * Z.ravel()  # per-feature contribution
    return feature_names, contrib

def contribution_perm_importance(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 5, random_state: int = 42):
    # For any model; returns permutation importances on the training set
    pre: ColumnTransformer = pipe.named_steps["pre"]
    model = pipe.named_steps["model"]
    Xt = pre.transform(X)
    r = permutation_importance(model, Xt, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    # No names here (post-transform). We'll map to names we built earlier.
    return r

def draw_contribution(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, feature_names: list[str]):
    # Try linear-style contributions first; fallback to permutation importance
    names, contrib = contribution_from_linear(pipe, X, feature_names)
    if names is not None and contrib is not None:
        order = np.argsort(np.abs(contrib))[::-1]
        names = np.array(names)[order][:25]
        contrib = np.array(contrib)[order][:25]
        fig, ax = plt.subplots(figsize=(8,6))
        sns.barplot(x=contrib, y=names, ax=ax)
        ax.set_title("Contribution chart (coeff * value) — top features")
        st.pyplot(fig)
        return

    # fallback: permutation importance
    st.caption("Linear-style contribution unavailable; showing permutation importance instead.")
    r = contribution_perm_importance(pipe, X, y)
    order = np.argsort(r.importances_mean)[::-1][:25]
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=r.importances_mean[order], y=np.array(feature_names)[order], ax=ax)
    ax.set_title("Permutation importance — top features")
    st.pyplot(fig)

if uploaded is None:
    st.info("Upload a CSV from the sidebar to begin.")
    st.stop()

# Load data
df = read_csv(uploaded)
st.success(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

all_cols = list(df.columns)
target = st.selectbox("Target column (y)", options=all_cols, index=max(0, all_cols.index("NewVolSales")) if "NewVolSales" in all_cols else 0)

# Train/eval
if chosen_models:
    run = st.button("Train & Evaluate Models", type="primary")
else:
    st.warning("Select at least one model to evaluate.")
    run = False

res_df = None
fitted = {}
split = None
feature_names_cache = None

if run:
    models = {k: model_choices[k] for k in chosen_models}
    with st.spinner("Training models..."):
        res_df, fitted, split = train_and_eval(df, target, models, test_size, random_state, auto_numeric_only)
    st.subheader("Results")
    st.dataframe(res_df, use_container_width=True)
    best = res_df.iloc[0]["Model"]
    st.success(f"Best model by MSE: **{best}**")
    # Download metrics
    csv = res_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download metrics CSV", data=csv, file_name="model_metrics.csv", mime="text/csv")

st.markdown("---")
st.header("Graphs")

graph_type = st.selectbox(
    "Graph type",
    [
        "Correlation heatmap",
        "Box plot",
        "Scatter plot",
        "Contribution chart (model-based)",
        "Categorical bar"
    ]
)

if graph_type == "Correlation heatmap":
    draw_corr(df)

elif graph_type == "Box plot":
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
    by_col = st.selectbox("Group by (categorical, optional)", options=["(none)"] + [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])])
    chosen = st.multiselect("Numeric columns", options=numeric_cols, default=numeric_cols[:1])
    by_col = None if by_col == "(none)" else by_col
    draw_box(df, chosen, by_col)

elif graph_type == "Scatter plot":
    x = st.selectbox("X", options=[c for c in df.columns if c != target])
    y = st.selectbox("Y", options=[target] + [c for c in df.columns if c != x])
    hue_opt = ["(none)"] + [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c]) and c not in [x, y]]
    hue = st.selectbox("Hue (optional categorical)", options=hue_opt)
    hue = None if hue == "(none)" else hue
    draw_scatter(df, x, y, hue)

elif graph_type == "Contribution chart (model-based)":
    if not fitted or split is None:
        st.info("Train at least one model first (click 'Train & Evaluate Models').")
    else:
        model_name = st.selectbox("Pick a trained model", options=list(fitted.keys()))
        pipe = fitted[model_name]
        X_train, X_test, y_train, y_test = split
        # fit the preprocessor on full X to get feature names
        pre: ColumnTransformer = pipe.named_steps["pre"]
        # Ensure pre is fitted (it is, via pipe.fit)
        feature_names = get_feature_names(pre, pd.concat([X_train, X_test], axis=0), target)
        draw_contribution(pipe, pd.concat([X_train, X_test], axis=0), np.concatenate([y_train, y_test]), feature_names)

elif graph_type == "Categorical bar":
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    cat = st.selectbox("Categorical column", options=cat_cols if cat_cols else ["(none)"])
    if cat_cols:
        agg_fn = st.selectbox("Aggregation", options=["count", "mean", "sum", "median"])
        agg_on = None
        if agg_fn != "count":
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            agg_on = st.selectbox("Numeric column to aggregate", options=num_cols if num_cols else ["(none)"])
            if not num_cols:
                agg_fn = "count"
                agg_on = None
        draw_categorical_bar(df, cat, agg_on, agg_fn)
    else:
        st.info("No categorical columns found in your data.")

st.markdown("---")
st.caption("Tip: Use the sidebar to switch models or tweak split/random state. Download metrics via the button above.")
