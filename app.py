import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

st.title("📊 Dashboard Prediksi Kategori Penjualan - Decision Tree C4.5")

# Upload file CSV
uploaded_file = st.file_uploader("Unggah Dataset CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Tabel Data")
    st.dataframe(df)

    # Mapping nama kolom agar konsisten
    df = df.rename(columns=lambda x: x.strip())

    # Visualisasi: Diagram Batang per Kategori Penjualan (Jumlah)
    st.subheader("📊 Jumlah Data per Kategori Penjualan")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Kategori Penjualan", order=["Laris", "Sedang", "Tidak Laris"], ax=ax1)
    st.pyplot(fig1)

    # Visualisasi: Produk Terlaris per Kategori (berdasarkan Qty)
    st.subheader("🏆 Produk Terlaris per Kategori")
    for kategori in ["Laris", "Sedang", "Tidak Laris"]:
        st.markdown(f"### {kategori}")
        top_produk = df[df["Kategori Penjualan"] == kategori].sort_values(by="Qty", ascending=False).head(5)
        fig2, ax2 = plt.subplots()
        sns.barplot(x="Qty", y="Nama Barang", data=top_produk, ax=ax2, palette="viridis")
        ax2.set_title(f"Top 5 Produk - {kategori}")
        st.pyplot(fig2)

    # Preprocessing untuk model
    st.subheader("⚙️ Pelatihan Model C4.5")

    # Fitur numerik
    fitur = ['Qty', 'Harga', 'Jual (Rupiah)']
    X = df[fitur]
    y = df['Kategori Penjualan']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Decision Tree (C4.5)
    model_c45 = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model_c45.fit(X_train, y_train)

    # Evaluasi Model
    y_pred = model_c45.predict(X_test)

    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.text("Akurasi Model:")
    st.write(f"{accuracy_score(y_test, y_pred):.2f}")

    st.subheader("📌 Confusion Matrix")
    fig3, ax3 = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=model_c45.classes_,
                yticklabels=model_c45.classes_,
                ax=ax3)
    st.pyplot(fig3)

    # Visualisasi Pohon Keputusan
    st.subheader("🌳 Visualisasi Pohon Keputusan")
    fig4, ax4 = plt.subplots(figsize=(16, 6))
    plot_tree(model_c45, feature_names=fitur, class_names=model_c45.classes_,
              filled=True, rounded=True, fontsize=10, ax=ax4)
    st.pyplot(fig4)

    # Visualisasi Tambahan: Distribusi Harga vs Qty
    st.subheader("📉 Distribusi Harga vs Jumlah (Qty)")
    fig5, ax5 = plt.subplots()
    sns.scatterplot(data=df, x='Harga', y='Qty', hue='Kategori Penjualan', palette='deep', ax=ax5)
    ax5.set_title("Harga vs Qty per Kategori Penjualan")
    st.pyplot(fig5)
