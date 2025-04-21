import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Step 1: Load datasets
diabetes_dataset = pd.read_csv('/kaggle/input/diabetes/train.csv') 
diabetes_test_dataset = pd.read_csv('/kaggle/input/diabetes/test.csv')

# Step 2: Separate features and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Step 3: Scale features
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# ✅ Fix: drop 'id' before scaling test data
X_TEST = diabetes_test_dataset
X_TEST = scaler.transform(X_TEST)

# Step 4: Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

# Step 5: Train the model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Step 6: Accuracy check
X_train_prediction = classifier.predict(X_train)
X_val_prediction = classifier.predict(X_val)

print('🔥 Training Accuracy:', accuracy_score(Y_train, X_train_prediction))
print('🧪 Validation Accuracy:', accuracy_score(Y_val, X_val_prediction))

# Step 7: Predict on actual test.csv
freeze_test = classifier.predict(X_TEST)
print("🧊 Test Predictions:", freeze_test[:10])  # First 10 preds

# Step 8: Output CSV
output = pd.DataFrame({
    'id': diabetes_test_dataset['id'],
    'Prediction': freeze_test
})

output.to_csv('svm_test_predictions.csv', index=False)
print("✅ Exported to svm_test_predictions.csv")
print(output.head())
