# Marketing-Mix-Model
MMM


About the Dataset
The data contains the sales data for two consecutive years of a particular product of some brand.

Each row contains the Volume of Sales for a week and different campaigning information or various promotion methods for that product information for each week for two consecutive years.

The information about that product and the years to which this data belongs are unknown.

#python create env
##Requirements
```bash
python3 -m venv venv
source venv/bin/activate
```

# Marketing Mix Model Evaluation (UI)

## 📦 Requirements
```bash
pip install streamlit pandas numpy scikit-learn xgboost matplotlib seaborn
```

## ▶️ Run
```bash
streamlit run app.py
```

## 📋 Usage
1. Upload your test dataset (CSV).
2. Select the target column (e.g., `NewVolSales`).
3. Choose models to evaluate.
4. View metrics (MSE, R², MAE, MAPE).
5. The app highlights the best model.
