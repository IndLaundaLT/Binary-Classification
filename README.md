# 🧠 Logistic Regression Binary Classifier - Breast Cancer Dataset

This project is part of an AI & ML internship and focuses on building a **binary classification model** using **Logistic Regression**.

---

## 📌 Objective

To train a logistic regression model that can classify tumors as **benign (0)** or **malignant (1)** based on various cell features from the **Breast Cancer Wisconsin Dataset**.

---

## 🛠️ Tools and Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 📁 Dataset

- **Source**: [Breast Cancer Wisconsin (Original)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- The dataset includes 9 predictive features (e.g., clump thickness, mitoses) and a target column `Class` (0 = benign, 1 = malignant).

---

## 🚀 Steps Followed

1. **Data Loading and Cleaning**
   - Loaded CSV data into a DataFrame
   - Removed any missing/null values

2. **Feature Engineering**
   - Extracted features (X) and labels (y)
   - Standardized features using `StandardScaler`

3. **Train/Test Split**
   - 80% training, 20% testing using `train_test_split`

4. **Model Training**
   - Trained a `LogisticRegression` model from scikit-learn

5. **Evaluation**
   - Confusion Matrix
   - Precision, Recall
   - ROC-AUC Score
   - ROC Curve plot

6. **Threshold Tuning**
   - Predicted probabilities
   - Tested a custom threshold to improve recall or precision

---

## 📊 Evaluation Metrics

- **Confusion Matrix**  
  ![confusion-matrix](screenshots/confusion_matrix.png)

- **ROC Curve**  
  ![roc-curve](screenshots/roc_curve.png)

- **ROC AUC Score**: *0.98 (example)*  
- **Precision**: *0.96*  
- **Recall**: *0.94*

---

## 🧠 Key Concepts

- **Sigmoid Function**: Maps real-valued input to a probability between 0 and 1.
- **Precision vs Recall**: Tradeoff depending on whether false positives or false negatives are more costly.
- **Threshold Tuning**: Improves metric performance based on the use case (e.g., medical diagnosis).
- **ROC-AUC**: Measures how well the model distinguishes between classes across all thresholds.

---

## 📌 Folder Structure

