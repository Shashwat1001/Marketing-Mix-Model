import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from sklearn.neural_network import MLPRegressor

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    has_xgb = True
except Exception:
    has_xgb = False
    st.warning("⚠️ XGBoost not available. Skipping it.")

# --- Define available models ---
models = {
    "SVR": SVR(kernel="rbf"),
    "AdaBoost": AdaBoostRegressor(),
    "RandomForest": RandomForestRegressor(),
    "DecisionTree": DecisionTreeRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "KNN": neighbors.KNeighborsRegressor(),
    "MLP": MLPRegressor()
}
if has_xgb:
    models["XGBoost"] = XGBRegressor()


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(abs((y_pred-y_test)/y_test))*100
    return {"MSE": mse, "R2": r2, "MAE": mae, "MAPE": mape}

st.title("📊 Marketing Mix Model Evaluation")

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    target = st.selectbox("Select Target Column", data.columns)
    features = [c for c in data.columns if c != target]

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model_choice = st.multiselect("Select Models to Evaluate", list(models.keys()))

    if st.button("Run Evaluation"):
        results = {}
        for name in model_choice:
            results[name] = evaluate_model(models[name], X_train, X_test, y_train, y_test)

        st.write("### Results")
        st.dataframe(pd.DataFrame(results).T)

        best_model = min(results, key=lambda x: results[x]["MSE"])
        st.success(f"Best model: {best_model} with lowest MSE")
