#  Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_score, recall_score
)

#  Step 2: Load Dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv"
data = pd.read_csv(url)
data = data.dropna(axis=1)  # Drop missing columns if any
print(data.head())

#  Step 3: Preprocessing
X = data.drop(['Class'], axis=1)
y = data['Class']  # Already binary: 0 (benign), 1 (malignant)


# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Step 4: Train Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

#  Step 5: Evaluation
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

#  Step 6: Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

#  Step 7: Threshold Tuning
threshold = 0.3  # Example custom threshold
y_pred_custom = (y_proba > threshold).astype(int)

print(f"\nCustom Threshold = {threshold}")
print("Precision:", precision_score(y_test, y_pred_custom))
print("Recall:", recall_score(y_test, y_pred_custom))
