import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
diabetes_dataset = pd.read_csv('/kaggle/input/diabetes/train.csv')
diabetes_test_dataset= pd.read_csv('/kaggle/input/diabetes/test.csv')

# Step 2: Separate features and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']
X_TEST=diabetes_test_dataset
# Step 3: Feature scaling
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_TEST = scaler.transform(diabetes_test_dataset)

# Step 4: Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Step 5: Train RandomForestClassifier
model = RandomForestClassifier(
    n_estimators=200,  # number of trees
    max_depth=10,       # prevent overfitting
    random_state=2
)

model.fit(X_train, Y_train)

# Step 6: Evaluate on training data
train_preds = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_preds)
print("🔥 Training Accuracy (RF):", train_accuracy)

# Step 7: Evaluate on test data
test_preds = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_preds)
print("🧪 Test Accuracy (RF):", test_accuracy)

freeze_test = model.predict(X_TEST)
print("🧊 Test Predictions:", freeze_test)  # printing first 10 predictions

output = pd.DataFrame({
    'id': diabetes_test_dataset['id'],
    'Prediction': freeze_test
})

output = pd.DataFrame({
    'id': diabetes_test_dataset['id'],
    'Prediction': freeze_test
})

output.to_csv('rf_test_predictions.csv', index=False)
print("✅ Exported to rf_test_predictions.csv")
print(output.head())

