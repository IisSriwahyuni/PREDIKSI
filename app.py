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
    X = df.select_dtypes(include=[np.number])  # semua kolom numerik
    y = df['Kategori Penjualan']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    st.pyplot(fig)
    
    # Skor evaluasi
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    
    st.subheader("📊 Skor Evaluasi")
    scores = {
        "Akurasi": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    # Tabel skor evaluasi
    st.table({k: [f"{v*100:.2f}%"] for k, v in scores.items()})
    
    # Diagram batang skor evaluasi
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(scores.keys())
    values = list(scores.values())
    sns.barplot(x=labels, y=values, palette="Set2", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Skor")
    ax.set_title("Visualisasi Skor Evaluasi")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')
    st.pyplot(fig)

# === K-Fold Cross Validation ===
elif menu == "K-Fold":
    st.header("🔄 K-Fold Cross Validation (5-Fold)")
    X = df[['Qty', 'Harga']]
    y = df['Kategori Penjualan']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    st.write("Akurasi per Fold:", scores)
    st.write(f"Rata-rata Akurasi: {scores.mean():.4f}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, len(scores)+1), scores, color="skyblue")
    ax.set_xlabel("Fold ke-")
    ax.set_ylabel("Akurasi")
    ax.set_ylim(0, 1)
    for i, v in enumerate(scores):
        ax.text(i+1, v + 0.02, f"{v*100:.1f}%", ha='center')
    st.pyplot(fig)

# === Visualisasi Pohon Keputusan ===
elif menu == "Decision Tree":
    st.header("🌳 Visualisasi Pohon Keputusan (Decision Tree C4.5)")
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(
        model,
        feature_names = X_train.columns,
        class_names=model.classes_,
        filled=True,
        rounded=True,
        fontsize=10
    )
    st.pyplot(fig)


