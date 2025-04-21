# 🩺 Diabetes Prediction Project

A machine learning project that predicts whether a person is diabetic or not based on diagnostic medical features. Built using the PIMA Indian Diabetes Dataset.

---

## 📌 Overview

This project uses:
- **Support Vector Machine (SVM) with Linear Kernel**
- **Random Forest Classifier**
- **XGBoost Classifier**

All models were trained and evaluated to predict the `Outcome` (0 = Non-diabetic, 1 = Diabetic). Feature scaling was applied using `StandardScaler`.

---

## 🔧 Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Jupyter / Kaggle Notebooks  

---

## 🗂️ Dataset

- **Train Dataset:** `train.csv` — Contains features and labels  
- **Test Dataset:** `test.csv` — Used to generate predictions for submission  

---

## 📊 Features Used

- `Pregnancies`  
- `Glucose`  
- `BloodPressure`  
- `SkinThickness`  
- `Insulin`  
- `BMI`  
- `DiabetesPedigreeFunction`  
- `Age`  

---

## 🧠 Models

### 1. SVM (Support Vector Machine)
- Linear kernel used for classification  
- Standard scaling applied  

### 2. Random Forest Classifier
- Ensemble learning using decision trees  
- Used for comparison with SVM  

### 3. XGBoost Classifier
- Gradient boosting framework  
- Efficient and robust to overfitting  

---

## ✅ Evaluation

| Model            | Training Accuracy | Validation Accuracy |
|------------------|-------------------|----------------------|
| SVM (Linear)     | ~100%             | ~78%                 |
| Random Forest    | ~100%             | ~80%                 |
| XGBoost          | ~100%             | ~79.2%               |

> Note: Accuracy may vary slightly depending on random seed.

---

## 📁 Output

Final test predictions were exported in the format:

```csv
id,Prediction
0,1
1,0
2,0
...
