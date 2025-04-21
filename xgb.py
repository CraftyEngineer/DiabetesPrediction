import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Step 1: Load the datasets
diabetes_dataset = pd.read_csv('/kaggle/input/diabetes/train.csv')
diabetes_test_dataset = pd.read_csv('/kaggle/input/diabetes/test.csv')

# Step 2: Separate features and labels from training data
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Step 3: Feature scaling
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Also scale the test set using the same scaler (important!)
X_TEST = scaler.transform(diabetes_test_dataset)

# Step 4: Train-test split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Step 5: Train XGBoost Classifier
model = XGBClassifier(
    n_estimators=100,
    max_depth=15,
    learning_rate=0.1,
    reg_lambda=1,
    reg_alpha=0,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=2
)
model.fit(X_train, Y_train)

# Step 6: Evaluate on training and validation data
train_preds = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_preds)
print("🔥 Training Accuracy:", train_accuracy)

val_preds = model.predict(X_val)
val_accuracy = accuracy_score(Y_val, val_preds)
print("🧪 Validation Accuracy:", val_accuracy)

# Step 7: Predict on test.csv
freeze_test = model.predict(X_TEST)
print("🧊 Test Predictions:", freeze_test)  # printing first 10 predictions

output = pd.DataFrame({
    'id': diabetes_test_dataset['id'],
    'Prediction': freeze_test
})

# Save to CSV
output.to_csv('xgb_test_predictions.csv', index=False)

# Preview
print("📁 Final Output:")
print(output.head())
