# Customer Churn Prediction Pipeline
# A professional, production-ready machine learning project

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Create output folder for visualizations
os.makedirs("plots", exist_ok=True)

# -------------------------------------------------------------
# 1. Load and Clean Dataset
# -------------------------------------------------------------
print("--- Step 1: Loading & Cleaning Data ---")
data = pd.read_csv("telecom_churn.csv")
print(f"Dataset shape: {data.shape}")

# TotalCharges might have blank spaces for new customers (tenure = 0)
# We coerce it to numeric, which puts NaN in place of spaces, then fill NaN with 0
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"] = data["TotalCharges"].fillna(0.0)

# Convert target variable 'Churn' (Yes/No) to binary (1/0)
data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

# Drop customerID as it is not a predictive feature
X = data.drop(columns=["customerID", "Churn"])
y = data["Churn"]

# -------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# -------------------------------------------------------------
print("\n--- Step 2: Generating EDA Plots (saved in 'plots/' directory) ---")

# Plot 1: Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="Churn", data=data, palette="Set2")
plt.title("Overall Customer Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots/churn_distribution.png")
plt.close()

# Plot 2: Tenure Distribution by Churn Status
plt.figure(figsize=(8, 5))
sns.kdeplot(data=data, x="tenure", hue="Churn", fill=True, common_norm=False, palette="Set1", alpha=0.5)
plt.title("Customer Tenure Distribution by Churn Status")
plt.xlabel("Tenure (Months)")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("plots/tenure_vs_churn.png")
plt.close()

# Plot 3: Monthly Charges vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=data, palette="Set3")
plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Monthly Charges ($)")
plt.tight_layout()
plt.savefig("plots/monthly_charges_vs_churn.png")
plt.close()

print("EDA plots saved successfully!")

# -------------------------------------------------------------
# 3. Train-Test Split
# -------------------------------------------------------------
# Using stratify=y ensures both training and testing sets have the same ratio of churned customers
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size: {X_test.shape[0]} rows")

# -------------------------------------------------------------
# 4. Preprocessing Pipeline
# -------------------------------------------------------------
# Identify numerical and categorical columns
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [
    col for col in X.columns if col not in numeric_features
]

print(f"\nNumerical features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# Build transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first", handle_unknown="error")

# Combine them using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -------------------------------------------------------------
# 5. Model Training & Comparison
# -------------------------------------------------------------
print("\n--- Step 3: Model Training & Evaluation ---")

# Define pipelines for the models we want to compare
pipelines = {
    "Logistic Regression": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
}

best_model_name = None
best_model_score = 0.0
best_pipeline = None

for name, pipeline in pipelines.items():
    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Test Accuracy: {acc:.4f}")
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))
    
    if acc > best_model_score:
        best_model_score = acc
        best_model_name = name
        best_pipeline = pipeline

print(f"\n>>> Best performing model: {best_model_name} with {best_model_score:.4f} accuracy.")

# -------------------------------------------------------------
# 6. Post-Evaluation: Confusion Matrix & Feature Importance
# -------------------------------------------------------------
# Plot Confusion Matrix for the best model
y_best_pred = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_best_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed", "Churned"])
disp.plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.savefig("plots/best_model_confusion_matrix.png")
plt.close()
print("Confusion Matrix plot saved to 'plots/best_model_confusion_matrix.png'")

# Plot Feature Importance (only for Random Forest)
if "Random Forest" in pipelines:
    rf_pipeline = pipelines["Random Forest"]
    # Get feature names after preprocessing
    ohe_cols = rf_pipeline.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(categorical_features)
    feature_names = list(numeric_features) + list(ohe_cols)
    
    # Get feature importances
    importances = rf_pipeline.named_steps["classifier"].feature_importances_
    
    # Sort and plot
    indices = np.argsort(importances)[::-1]
    # Limit to top 15 features for clarity
    top_n = min(15, len(feature_names))
    
    plt.figure(figsize=(10, 6))
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.bar(range(top_n), importances[indices[:top_n]], align="center", color="skyblue")
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("plots/feature_importances.png")
    plt.close()
    print("Feature Importances plot saved to 'plots/feature_importances.png'")

# -------------------------------------------------------------
# 7. Predicting for a New Customer
# -------------------------------------------------------------
print("\n--- Step 4: Testing Prediction with a New Customer ---")
# Example of a new customer (raw dict matching CSV headers)
new_cust_data = {
    "gender": ["Female"],
    "SeniorCitizen": [0],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "tenure": [3],             # Low tenure
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["Fiber optic"], # Fiber optic (high risk)
    "OnlineSecurity": ["No"],           # No security (high risk)
    "OnlineBackup": ["No"],
    "DeviceProtection": ["No"],
    "TechSupport": ["No"],              # No tech support (high risk)
    "StreamingTV": ["Yes"],
    "StreamingMovies": ["Yes"],
    "Contract": ["Month-to-month"],     # Month-to-month (high risk)
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [95.00],          # High monthly charge (high risk)
    "TotalCharges": [285.00]
}

new_cust_df = pd.DataFrame(new_cust_data)

# Predict churn
pred = best_pipeline.predict(new_cust_df)[0]
pred_prob = best_pipeline.predict_proba(new_cust_df)[0][1]

print("New Customer Data:")
for k, v in new_cust_data.items():
    print(f"  {k}: {v[0]}")

print(f"\nPrediction using {best_model_name}:")
if pred == 1:
    print(f"  Result: Customer will CHURN [!] (Probability: {pred_prob:.2%})")
else:
    print(f"  Result: Customer will STAY [OK] (Probability of Churn: {pred_prob:.2%})")
