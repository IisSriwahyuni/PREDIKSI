import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Judul Aplikasi
st.title("Dashboard Prediksi Kategori Penjualan Produk Wings")

# Upload File CSV
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file is not None:
    # Baca file CSV
    df = pd.read_csv(uploaded_file)

    # Tampilkan Dataframe
    st.subheader("Dataset")
    st.dataframe(df.head())

    # Informasi kolom
    if 'Kategori Penjualan' in df.columns and 'Qty' in df.columns:
        st.subheader("Jumlah Produk per Kategori Penjualan (Diurutkan berdasarkan Qty)")
        kategori_terurut = df.groupby('Kategori Penjualan').apply(lambda x: x.sort_values(by='Qty', ascending=False)).reset_index(drop=True)
        st.dataframe(kategori_terurut[['Nama Barang', 'Qty', 'Kategori Penjualan']])

    # Preprocessing Data
    st.subheader("Preprocessing dan Training Model")
    try:
        fitur = ['Qty', 'Harga']
        X = df[fitur]
        y = df['Kategori Penjualan']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Buat model C4.5 (Decision Tree dengan entropy)
        model_c45 = DecisionTreeClassifier(criterion='entropy', random_state=42)
        model_c45.fit(X_train, y_train)

        # Prediksi
        y_pred = model_c45.predict(X_test)

        # Evaluasi
        st.markdown("### Evaluasi Model")
        st.text("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.text(cm)

        st.text("\nClassification Report:")
        cr = classification_report(y_test, y_pred)
        st.text(cr)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"Akurasi Model: {acc:.2f}")

        # Visualisasi Confusion Matrix
        st.subheader("Visualisasi Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model_c45.classes_, yticklabels=model_c45.classes_, ax=ax)
        plt.xlabel("Prediksi")
        plt.ylabel("Aktual")
        plt.title("Confusion Matrix - Decision Tree C4.5")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Terjadi error saat memproses data atau melatih model: {e}")
