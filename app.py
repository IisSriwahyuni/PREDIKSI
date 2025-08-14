import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score,
    recall_score, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
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
    "Dataset", "Confusion Matrix (Manual)", "Eval Model (Auto)",
    "ROC-AUC", "K-Fold"
])

# === Tampilkan Dataset ===
if menu == "Dataset":
    st.header("📁 Data Penjualan Produk Wings")
    st.dataframe(df)

# === Confusion Matrix Manual ===
elif menu == "Confusion Matrix (Manual)":
    st.header("📌 Confusion Matrix — Sesuai Evaluasi Terakhir")
    cm = np.array([[29, 2, 0], [0, 17, 0], [0, 0, 44]])
    classes = ['Laris', 'Sedang', 'Tidak Laris']
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    st.pyplot(fig)

    report = {
        "Class": classes + ["accuracy", "macro avg", "weighted avg"],
        "precision": [1.00, 0.89, 1.00, "", 0.96, 0.98],
        "recall":    [0.94, 1.00, 1.00, "", 0.98, 0.98],
        "f1-score":  [0.97, 0.94, 1.00, "", 0.97, 0.98],
        "support":   [31, 17, 44, 92, 92, 92]
    }
    st.subheader("📋 Classification Report")
    st.dataframe(pd.DataFrame(report).set_index("Class"))

    acc = 0.9782608695652174
    st.metric("🎯 Akurasi Model", f"{acc * 100:.2f}%")

# === Evaluasi Otomatis dari Model ===
elif menu == "Eval Model (Auto)":
    st.subheader("Skor Evaluasi")
    fig, ax = plt.subplots(figsize=(6, 4))
    scores = [precision, recall, f1]
    labels = ["Precision", "Recall", "F1 Score"]
    sns.barplot(x=labels, y=scores, palette="Set2", ax=ax)
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(scores):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')
    st.pyplot(fig)

# === ROC-AUC Multi-Class ===
elif menu == "ROC-AUC":
    st.header("📈 ROC-AUC Multi-Class")
    y_prob = model.predict_proba(X_test)
    classes = model.classes_
    y_bin = label_binarize(y_test, classes=classes)
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ['blue', 'green', 'red']
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"{cls} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
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

