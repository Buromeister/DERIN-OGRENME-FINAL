import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

veri_seti = pd.read_csv('veri_seti.csv')

# Girdi (X) ve Çıktı (y) değişkenlerini ayırma
X = veri_seti[['kac_hamile', 'glukoz_test', 'kan_basinci', 'deri_kalinlik', 'insulin_seviye', 'vucut_kitle', 'aile_diyabet', 'h_yaş']]
y = veri_seti['sinif_degiskeni']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Eğitim seti boyutu:", X_train.shape)
print("Test seti boyutu:", X_test.shape)

mlp = MLPClassifier(random_state=42)
mlp.fit(X_train, y_train)

y_pred_mlp = mlp.predict(X_test)
y_prob_mlp = mlp.predict_proba(X_test)[:, 1]

print("MLP Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_mlp))
print("MLP Konfüzyon Matrisi:\n", confusion_matrix(y_test, y_pred_mlp))

fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
y_prob_svm = svm.predict_proba(X_test)[:, 1]

print("SVM Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_svm))
print("SVM Konfüzyon Matrisi:\n", confusion_matrix(y_test, y_pred_svm))

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure()
plt.plot(fpr_mlp, tpr_mlp, color='darkorange', lw=2, label='MLP ROC curve (area = %0.2f)' % roc_auc_mlp)
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label='SVM ROC curve (area = %0.2f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()