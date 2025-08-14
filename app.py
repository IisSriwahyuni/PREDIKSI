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

# Konfigurasi tampilan halaman
st.set_page_config(layout="wide")
st.title("📊 Dashboard Prediksi & Evaluasi Model Decision Tree (C4.5)")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("PRODUK_WINGS_YMART_BERSIH.csv")

df = load_data()

# Validasi kolom
required_columns = ['Qty', 'Harga', 'Kategori Penjualan']
if not all(col in df.columns for col in required_columns):
    st.error("❌ Dataset tidak lengkap. Harus mengandung kolom: Qty, Harga, Kategori Penjualan.")
    st.stop()

# Sidebar Navigasi
menu = st.sidebar.radio("NAVIGASI", [
    "Dataset", "Distribusi Penjualan", "Pola Penjualan",
    "Decision Tree", "Evaluasi Model", "ROC-AUC", "K-Fold", "Visualisasi Tambahan"
])

# Dataset
if menu == "Dataset":
    st.header("📁 Data Penjualan Produk Wings")
    st.dataframe(df)

# Distribusi Penjualan
elif menu == "Distribusi Penjualan":
    st.header("📊 Distribusi Qty Penjualan")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(df["Qty"], bins=30, kde=True, color="skyblue", ax=ax)
    st.pyplot(fig)

# Pola Penjualan
elif menu == "Pola Penjualan":
    st.header("📈 Pola Penjualan per Kategori")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.boxplot(x="Kategori Penjualan", y="Qty", data=df, palette="viridis", ax=ax)
    st.pyplot(fig)

# Decision Tree
elif menu == "Decision Tree":
    st.header("🌳 Visualisasi Decision Tree")
    X = df[['Qty', 'Harga']]
    y = df['Kategori Penjualan']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(model, feature_names=['Qty', 'Harga'], class_names=model.classes_, filled=True, rounded=True, fontsize=10, ax=ax)
    st.pyplot(fig)

# Evaluasi Model
elif menu == "Evaluasi Model":
    st.header("📋 Evaluasi Model dengan Data Uji")
    X = df[['Qty', 'Harga']]
    y = df['Kategori Penjualan']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    cr = classification_report(y_test, y_pred, target_names=model.classes_, output_dict=True)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.subheader("📌 Confusion Matrix")
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.subheader("📌 Classification Report")
    st.dataframe(pd.DataFrame(cr).transpose().round(2))

    st.subheader("📌 Skor Evaluasi")
    fig, ax = plt.subplots(figsize=(4, 3))
    scores = [precision, recall, f1]
    labels = ["Precision", "Recall", "F1 Score"]
    sns.barplot(x=labels, y=scores, palette="Set2", ax=ax)
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(scores):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontsize=8)
    st.pyplot(fig)

# ROC-AUC
elif menu == "ROC-AUC":
    st.header("📈 ROC-AUC Multi-Class")
    X = df[['Qty', 'Harga']]
    y = df['Kategori Penjualan']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)

    classes = model.classes_
    y_bin = label_binarize(y_test, classes=classes)

    fig, ax = plt.subplots()
    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_label} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(fig)

# K-Fold
elif menu == "K-Fold":
    st.header("🔄 K-Fold Cross Validation")
    X = df[['Qty', 'Harga']]
    y = df['Kategori Penjualan']
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

    st.write("Akurasi per Fold:", scores)
    st.write(f"Rata-rata Akurasi: {scores.mean():.4f}")

    fig, ax = plt.subplots()
    ax.bar(range(1, len(scores) + 1), scores, color="skyblue")
    ax.set_xlabel("Fold ke-")
    ax.set_ylabel("Akurasi")
    ax.set_ylim(0, 1)
    for i, v in enumerate(scores):
        ax.text(i+1, v + 0.02, f"{v*100:.1f}%", ha='center', fontsize=8)
    st.pyplot(fig)

# Visualisasi Tambahan
elif menu == "Visualisasi Tambahan":
    st.header("📊 Visualisasi Tambahan")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(data=df, x='Kategori Penjualan', palette='coolwarm', ax=ax)
    st.pyplot(fig)
