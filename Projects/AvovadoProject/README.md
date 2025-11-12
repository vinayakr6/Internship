# Avocado Streamlit App

This project converts your Avocado Jupyter notebook into a Streamlit app.
It supports loading a bundled CSV (`data/avocado.csv`) or uploading a CSV, loading a pre-trained model (`model/model.pkl`) or training in-app, interactive EDA (Plotly), and single-row prediction.


| Category             | Description                                                                                                            |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **📂 Data Input**    | Choose between bundled dataset (`data/avocado.csv`) or uploading your own CSV.                                         |
| **🧮 Model Options** | Load a pre-trained model (`model/model.pkl`) or train in-app using RandomForestRegressor.                              |
| **⚙️ Interactivity** | Adjustable features, target variable, train/test split ratio, and hyperparameters (e.g., `n_estimators`, `max_depth`). |
| **📊 EDA (Plotly)**  | Interactive charts: price over time, price by region, histograms, and correlation heatmap.                             |
| **📈 Evaluation**    | Metrics: MSE, RMSE, MAE, R²; visualizations: Actual vs Predicted, Residuals vs Predicted.                              |
| **🔮 Prediction**    | Real-time single-row prediction form — enter values and get the predicted price instantly.                             |
| **💾 Persistence**   | Option to save or upload models directly in the interface.                                                             |
