# Avocado Streamlit App

This project converts your Avocado Jupyter notebook into a Streamlit app.
It supports loading a bundled CSV (`data/avocado.csv`) or uploading a CSV, loading a pre-trained model (`model/model.pkl`) or training in-app, interactive EDA (Plotly), and single-row prediction.

## Project structure
avocado-streamlit-app/
├─ app.py
├─ utils.py
├─ requirements.txt
├─ Dockerfile
├─ data/
│ └─ avocado.csv # optional bundled dataset
├─ model/
│ └─ model.pkl # optional pre-trained model
