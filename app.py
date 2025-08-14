import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# -------------------------
# 1. Load Dataset
# -------------------------
@st.cache_data
def load_data():
    # Ganti dengan dataset Anda
    df = pd.read_csv("dataset_wings.csv")
    return df

df = load_data()

st.title("📊 Dashboard Prediksi Penjualan Produk Wings")

# -------------------------
# 2. Preprocessing
# -------------------------
X = df.drop(columns=["Kategori Penjualan"])  # fitur
y = df["Kategori Penjualan"]  # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 3. Model
# -------------------------
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------------------
# 4. Pengujian Matriks
# -------------------------
st.header("📈 Pengujian Matriks")
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
report = classification_report(y_test, y_pred, output_dict=True)

# Tampilkan Confusion Matrix
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
st.pyplot(fig)

# Tampilkan metrik
st.write("**Classification Report**")
st.dataframe(pd.DataFrame(report).transpose())

# -------------------------
# 5. ROC & AUC
# -------------------------
st.header("📉 ROC & AUC")

# Konversi label menjadi numerik untuk ROC
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_bin = le.fit_transform(y)
y_train_bin = le.transform(y_train)
y_test_bin = le.transform(y_test)

# One-vs-rest ROC
y_score = model.predict_proba(X_test)
fpr, tpr, _ = roc_curve(y_test_bin, y_score[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
ax2.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend(loc="lower right")
st.pyplot(fig2)

# -------------------------
# 6. K-Fold Cross Validation
# -------------------------
st.header("🔄 K-Fold Cross Validation")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
st.write(f"**Rata-rata Akurasi:** {scores.mean():.2f}")
st.write(f"**Akurasi per Fold:** {scores}")

# -------------------------
# 7. Pohon Keputusan
# -------------------------
st.header("🌳 Visualisasi Pohon Keputusan")
fig3, ax3 = plt.subplots(figsize=(15, 8))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
st.pyplot(fig3)
