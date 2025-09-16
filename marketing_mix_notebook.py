
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

# --- Define available models ---
models = {
    "SVR": SVR(kernel="rbf"),
    "AdaBoost": AdaBoostRegressor(),
   # "XGBoost": XGBRegressor(),
    "RandomForest": RandomForestRegressor(),
    "DecisionTree": DecisionTreeRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "KNN": neighbors.KNeighborsRegressor(),
    "MLP": MLPRegressor()
}

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(abs((y_pred-y_test)/y_test))*100
    return {"MSE": mse, "R2": r2, "MAE": mae, "MAPE": mape}

# Example usage in notebook:
# data = pd.read_csv("mktmix.csv")
# target = "NewVolSales"
# X = data.drop(columns=[target])
# y = data[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# results = {name: evaluate_model(model, X_train, X_test, y_train, y_test) for name, model in models.items()}
# pd.DataFrame(results).T
