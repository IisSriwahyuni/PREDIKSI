import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, accuracy_score
)
import numpy as np

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Dashboard C4.5 Evaluasi", layout="wide")
st.title("📊 Dashboard Prediksi & Evaluasi Model Decision Tree (C4.5)")

# === Load Dataset ===
@st.cache_data
def load_data():
    return pd.read_csv("PRODUK_WINGS_YMART_BERSIH.csv")

df = load_data()
required_columns = ['Qty', 'Harga', 'Kategori Penjualan']
if df.empty or not all(col in df.columns for col in required_columns):
    st.error(f"Dataset tidak valid. Harus memiliki kolom: {', '.join(required_columns)}")
    st.stop()

# === Fungsi Training Sekali ===
@st.cache_resource
def train_model():
    X = df[['Qty', 'Harga']]
    y = df['Kategori Penjualan']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

model, X_train, X_test, y_train, y_test = train_model()

# === Navigasi Sidebar ===
menu = st.sidebar.radio("Navigation", [
    "Dataset", "Confusion Matrix", "K-Fold", "Decision Tree"
])

# === Tampilkan Dataset ===
if menu == "Dataset":
    st.header("📁 Data Penjualan Produk Wings")
    st.dataframe(df)

# === Confusion Matrix + Skor Evaluasi + Diagram Batang ===
elif menu == "Confusion Matrix":
    st.header("📌 Confusion Matrix & Skor Evaluasi Model")
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    classes = model.classes_
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blue
