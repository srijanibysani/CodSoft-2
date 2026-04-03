print("program Started")
import pandas as pd
data = pd.read_csv("Telco-Customer-Churn.csv")
print(data.head())
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset


data = pd.read_csv("Telco-Customer-Churn.csv")

print(data.columns)

# Remove spaces in column names
data.columns = data.columns.str.strip().str.replace(" ", "")

# Remove customerID column if present
if "customerID" in data.columns:
    data = data.drop('customerID', axis=1, errors='ignore')
# Convert TotalCharges to numeric if present
if "TotalCharges" in data.columns:
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

# Fill missing values
data = data.fillna(0)

# Encode text columns
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

# Separate features and target
X = data.drop("Exited", axis=1)
X = data.drop("Exited", axis=1)

X = X.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

X["Gender"] = X["Gender"].map({"Male":1, "Female":0})

X = pd.get_dummies(X, columns=["Geography"], drop_first=True)

y = data["Exited"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, pred))
# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
# ==========================
# Testing the Model
# ==========================

print("\n--- Testing on 5 customers from test set ---")
for i in range(5):
    customer = X_test.iloc[i].values.reshape(1, -1)
    prediction = model.predict(customer)[0]
    actual = y_test.iloc[i]
    print(f"Customer {i+1}: Predicted = {prediction}, Actual = {actual}")

print("\n--- Testing on a new customer ---")
# Example: [CreditScore, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_Germany, Geography_Spain]
new_customer = np.array([[650, 1, 40, 3, 50000, 2, 1, 1, 60000, 0, 1]])  
new_prediction = model.predict(new_customer)[0]
if new_prediction == 1:
    print("New Customer Prediction: Will Leave (Churn)")
else:
    print("New Customer Prediction: Will Stay (Not Churn)")


X_test_copy = X_test.copy()
X_test_copy['Actual'] = y_test
X_test_copy['Predicted'] = y_pred
print("\n--- First 5 rows of Test Set with Predictions ---")
print(X_test_copy.head(5))

print("\nDistribution of Actual Churn in test set:")
print(y_test.value_counts())

print("\nDistribution of Predicted Churn in test set:")
print(pd.Series(y_pred).value_counts())


X_test_copy = X_test.copy()
X_test_copy['Actual'] = y_test
X_test_copy['Predicted'] = y_pred

churned_customers = X_test_copy[X_test_copy['Actual'] == 1]
print("\nSome customers who actually left (Churned):")
print(churned_customers.head(5))

import numpy as np
new_customer = np.array([[580, 0, 25, 1, 80000, 1, 0, 0, 30000, 1, 0]])  # Example high-risk customer
new_prediction = model.predict(new_customer)[0]
print("\nNew Customer Prediction:", "Will Leave (Churn)" if new_prediction==1 else "Will Stay")


import matplotlib.pyplot as plt

plt.figure(figsize=(5,4))
plt.hist(y_test, alpha=0.5, label='Actual')
plt.hist(y_pred, alpha=0.5, label='Predicted')
plt.xlabel("Exited (0=Stay, 1=Leave)")
plt.ylabel("Number of Customers")
plt.legend()
plt.show()
