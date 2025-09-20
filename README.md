# Marketing Mix Model (Streamlit App)

This repository contains an interactive **Marketing Mix Modeling (MMM)** Streamlit application.  
It allows you to:

- Upload a dataset (CSV).
- Select a **target variable** and **features**.
- Train multiple ML models (Linear Regression, Ridge, Lasso, Random Forest, and optionally XGBoost).
- Evaluate models using **MSE, MAE, MAPE, R²**.
- Visualize results with multiple **graphs**:
  - Correlation heatmap
  - Box plot (grouped by categorical variable, optional)
  - Scatter plot (with optional hue)
  - Contribution chart (linear model contributions / permutation importance)
  - Categorical bar (count/mean/sum/median)

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/Marketing-Mix-Model.git
cd Marketing-Mix-Model
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📊 Usage

1. **Upload a CSV** file in the sidebar.  
   Example: dataset with columns like `TV`, `Radio`, `Social`, `Price`, `NewVolSales`.

2. **Choose target column** (e.g., `NewVolSales`).  

3. **Select algorithms** (Linear Regression, Ridge, Lasso, Random Forest, XGBoost).  

4. Click **Train & Evaluate Models**.  
   - Results table shows errors (MSE, MAE, MAPE, R²).  
   - Best model by MSE is highlighted.  
   - Download model metrics as CSV.

5. Open the **Graphs** section to generate:
   - **Correlation heatmap** → view numeric correlations.
   - **Box plot** → numeric columns grouped by categorical column.
   - **Scatter plot** → any two variables with optional categorical hue.
   - **Contribution chart** → feature importance/contributions from the trained model.
   - **Categorical bar** → frequency or aggregated statistics of categorical variables.

---

## 📂 Project Structure

```
Marketing-Mix-Model/
│── app.py              # Main Streamlit application
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
│── data/               # (Optional) Place sample datasets here
│── notebooks/          # Jupyter notebooks for exploration
```

---

## 🧾 Notes

- Contribution chart:
  - For linear models: coefficient × feature value (average row).
  - For non-linear models: permutation importance.
- Works with any tabular CSV data.
- Best used with marketing data: ad spend, impressions, sales, etc.

---

## 📸 Screenshots

*(Add screenshots of the app here after running locally)*

---

## ✨ Next Steps / Extensions
- Add stacked contribution over time (for weekly/date datasets).
- Export trained model as pickle file.
- Integrate dashboards with Power BI or Tableau.

---

## 👨‍💻 Author
Developed by **Shardul / Shashwat1001**  
