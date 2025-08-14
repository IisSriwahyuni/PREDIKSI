# =======================
# Dashboard Streamlit untuk Prediksi Penjualan Wings
# Menampilkan: Matriks Evaluasi, K-Fold Cross Validation, Pohon Keputusan
# =======================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# --------------------
# 1. Load Data
# --------------------
st.title("📊 Dashboard Prediksi Penjualan Produk Wings")
st.write("Model: Decision Tree (C4.5 Style)")

uploaded_file = st.file_uploader("Unggah dataset CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset")
    st.dataframe(df.head())

    # --------------------
    # 2. Persiapan Data
    # --------------------
    X = df.drop(columns=["Kategori Penjualan"])  # fitur
    y = df["Kategori Penjualan"]  # target

    # --------------------
    # 3. Split Data
    # --------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --------------------
    # 4. Training Model
    # --------------------
    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --------------------
    # 5. Evaluasi Matriks
    # --------------------
    st.subheader("📈 Evaluasi Matriks")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1-Score", f"{f1:.2f}")

    # Confusion Matrix
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax_cm)
    st.pyplot(fig_cm)

    # --------------------
    # 6. K-Fold Cross Validation
    # --------------------
    st.subheader("🔄 K-Fold Cross Validation")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf)

    st.write(f"Rata-rata Akurasi K-Fold: {cv_scores.mean():.2f}")
    st.write(f"Skor per fold: {cv_scores}")

    # Grafik K-Fold
    fig_kfold, ax_kfold = plt.subplots()
    ax_kfold.plot(range(1, len(cv_scores) + 1), cv_scores, marker='o')
    ax_kfold.set_title("Hasil K-Fold Cross Validation")
    ax_kfold.set_xlabel("Fold ke-")
    ax_kfold.set_ylabel("Akurasi")
    ax_kfold.grid(True)
    st.pyplot(fig_kfold)

    # --------------------
    # 7. Visualisasi Pohon Keputusan
    # --------------------
    st.subheader("🌳 Visualisasi Pohon Keputusan")
    fig_tree, ax_tree = plt.subplots(figsize=(12, 8))
    plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, ax=ax_tree)
    st.pyplot(fig_tree)
else:
    st.warning("Silakan unggah file dataset CSV terlebih dahulu.")
