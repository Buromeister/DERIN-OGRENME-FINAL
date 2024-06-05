import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

veri_seti = pd.read_csv("veri_seti.csv")

# Özellikler (X) ve hedef değişken (y) olarak ayır
X = veri_seti.drop('sinif_degiskeni', axis=1)
y = veri_seti['sinif_degiskeni']

# Eğitim ve test setlerini oluştur
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# K-en yakın komşuluk sınıflandırıcısını uygula
k_degerleri = range(1, 21)  # 1'den 20'ye kadar k değerlerini dene
dogruluklar = []

for k in k_degerleri:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_egitim, y_egitim)
    dogruluklar.append(knn.score(X_test, y_test))

en_iyi_k = k_degerleri[dogruluklar.index(max(dogruluklar))]
print("En iyi k değeri:", en_iyi_k)

en_iyi_model = KNeighborsClassifier(n_neighbors=en_iyi_k)
en_iyi_model.fit(X_egitim, y_egitim)
y_tahmin = en_iyi_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_tahmin))
print("\nClassification Report:\n", classification_report(y_test, y_tahmin))

y_proba = en_iyi_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()