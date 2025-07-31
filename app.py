# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("Prediksi Penjualan Produk Wings di Y-Mart Menggunakan Decision Tree C4.5")

# Load dataset
st.subheader("1. Data Penjualan")
uploaded_file = st.file_uploader("Unggah file CSV dataset", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.dataframe(df.head())

    # Pastikan kolom yang dibutuhkan ada
    required_columns = ['Qty', 'Harga', 'Jual', 'Kategori']
    if not all(col in df.columns for col in required_columns):
        st.error("Dataset harus memiliki kolom: 'Qty', 'Harga', 'Jual', dan 'Kategori'")
        st.stop()

    # Konversi kolom numerik
    for col in ['Qty', 'Harga', 'Jual']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Hapus data kosong
    df.dropna(subset=required_columns, inplace=True)

    # Label encoding untuk target
    label_encoder = LabelEncoder()
    df['Kategori'] = label_encoder.fit_transform(df['Kategori'])

    # Split fitur dan target
    X = df[['Qty', 'Harga', 'Jual']]
    y = df['Kategori']

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi model Decision Tree (C4.5 menggunakan entropy)
    model_c45 = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=42)
    model_c45.fit(X_train, y_train)

    # Prediksi dan evaluasi
    y_pred = model_c45.predict(X_test)

    # Evaluasi model
    st.subheader("2. Evaluasi Model C4.5")
    st.write("**Akurasi Model:**", f"{accuracy_score(y_test, y_pred):.2%}")
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    st.pyplot(fig_cm)

    # Visualisasi pohon keputusan
    st.subheader("3. Visualisasi Decision Tree (C4.5)")
    fig_tree, ax_tree = plt.subplots(figsize=(16, 8))
    plot_tree(
        model_c45,
        feature_names=X.columns,
        class_names=label_encoder.classes_,
        filled=True,
        rounded=True,
        fontsize=10
    )
    st.pyplot(fig_tree)

else:
    st.warning("Silakan unggah file dataset terlebih dahulu (.csv)")
