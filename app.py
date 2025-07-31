GITHUB

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# Konfigurasi tampilan halaman
st.set_page_config(layout="wide")
st.title("📊 Prediksi Penjualan Produk Wings Toko Ymart Kawali")

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
    "Dataset", "Distribusi Penjualan", "Pola Penjualan", "Decision Tree", "Evaluasi Model", "Visualisasi Tambahan"
])

# Tampilan Dataset
if menu == "Dataset":
    st.header("📁 Data Penjualan Produk Wings")
    st.dataframe(df)

# Distribusi Penjualan
elif menu == "Distribusi Penjualan":
    st.header("📊 Distribusi Qty Penjualan")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.histplot(df["Qty"], bins=30, kde=True, color="skyblue", ax=ax)
    ax.set_xlabel("Qty", fontsize=10)
    ax.set_ylabel("Jumlah Produk", fontsize=10)
    ax.set_title("Distribusi Qty Penjualan Produk", fontsize=12)
    st.pyplot(fig)

# Pola Penjualan
elif menu == "Pola Penjualan":
    st.header("📈 Pola Penjualan per Kategori")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.boxplot(x="Kategori Penjualan", y="Qty", data=df, palette="viridis", ax=ax)
    ax.set_title("Pola Penjualan per Kategori", fontsize=10)
    ax.set_xlabel("Kategori Penjualan", fontsize=9)
    ax.set_ylabel("Qty", fontsize=8)
    st.pyplot(fig)

# Decision Tree
elif menu == "Decision Tree":
    st.header("🌳 Visualisasi Decision Tree")
    X = df[['Qty', 'Harga']]
    y = df['Kategori Penjualan']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    st.markdown(f"""
    **Pembagian Data:**
    - Total Data: {len(df)}
    - Data Latih: {len(X_train)}
    - Data Uji: {len(X_test)}
    """)

    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(model,
              feature_names=['Qty', 'Harga'],
              class_names=model.classes_,
              filled=True,
              rounded=True,
              fontsize=10,
              ax=ax)
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

    # Confusion Matrix
    st.subheader("📌 Confusion Matrix")
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    st.dataframe(cm_df)

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix Heatmap")
    st.pyplot(fig)

    # Classification Report
    st.subheader("📌 Classification Report")
    cr_df = pd.DataFrame(cr).transpose()
    st.dataframe(cr_df.round(2))

    # Evaluasi Bar Plot
    st.subheader("📌 Metode Evaluasi Skor (Data Uji)")
    st.markdown(f"""
    - 🎯 **Precision (Weighted Avg)**: `{precision:.2f}`
    - 🎯 **Recall (Weighted Avg)**: `{recall:.2f}`
    - 🎯 **F1 Score (Weighted Avg)**: `{f1:.2f}`
    """)

    fig, ax = plt.subplots(figsize=(4, 3))
    scores = [precision, recall, f1]
    labels = ["Precision", "Recall", "F1 Score"]
    sns.barplot(x=labels, y=scores, palette="Set2", ax=ax)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Evaluasi Model (Data Uji)")
    for i, v in enumerate(scores):
        ax.text(i, v + 0.02, f"{v*100:.1f}%", ha='center', fontsize=8)
    st.pyplot(fig)

# Visualisasi Tambahan
elif menu == "Visualisasi Tambahan":
    st.header("📊 Visualisasi Tambahan")

    # Countplot kategori
    st.subheader("Jumlah Produk per Kategori Penjualan")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(data=df, x='Kategori Penjualan', palette='coolwarm', ax=ax)
    ax.set_title("Jumlah Data per Kategori Penjualan")
    st.pyplot(fig)

    # Scatter plot Qty vs Harga
    st.subheader("Sebaran Qty vs Harga berdasarkan Kategori")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.scatterplot(data=df, x='Qty', y='Harga', hue='Kategori Penjualan', palette='deep', ax=ax)
    ax.set_title("Qty vs Harga per Kategori Penjualan")
    st.pyplot(fig)

    # Pie chart proporsi kategori
    st.subheader("Proporsi Kategori Penjualan")
    fig, ax = plt.subplots(figsize=(3, 3))
    kategori_counts = df['Kategori Penjualan'].value_counts()
    if not kategori_counts.empty:
        ax.pie(kategori_counts, labels=kategori_counts.index, autopct='%1.1f%%', startangle=140)
        ax.axis('equal')
        st.pyplot(fig)
    else:
        st.warning("⚠️ Data kategori penjualan kosong, tidak dapat ditampilkan.")
