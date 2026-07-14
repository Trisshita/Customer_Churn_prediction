import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

n_samples = 5000

# Generate basic features
customer_ids = [f"CUST-{i:04d}" for i in range(1, n_samples + 1)]
genders = np.random.choice(["Female", "Male"], size=n_samples)
senior_citizens = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
partners = np.random.choice(["Yes", "No"], size=n_samples)
dependents = np.random.choice(["Yes", "No"], size=n_samples, p=[0.7, 0.3])

# Tenure: older customers have longer tenure
tenures = np.random.randint(0, 73, size=n_samples)

# Services
phone_service = np.random.choice(["Yes", "No"], size=n_samples, p=[0.9, 0.1])
multiple_lines = []
for p in phone_service:
    if p == "No":
        multiple_lines.append("No phone service")
    else:
        multiple_lines.append(np.random.choice(["No", "Yes"], p=[0.55, 0.45]))

internet_service = np.random.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.3, 0.45, 0.25])

online_security = []
online_backup = []
device_protection = []
tech_support = []
streaming_tv = []
streaming_movies = []

for i in internet_service:
    if i == "No":
        online_security.append("No internet service")
        online_backup.append("No internet service")
        device_protection.append("No internet service")
        tech_support.append("No internet service")
        streaming_tv.append("No internet service")
        streaming_movies.append("No internet service")
    else:
        online_security.append(np.random.choice(["No", "Yes"], p=[0.65, 0.35]))
        online_backup.append(np.random.choice(["No", "Yes"], p=[0.6, 0.4]))
        device_protection.append(np.random.choice(["No", "Yes"], p=[0.6, 0.4]))
        tech_support.append(np.random.choice(["No", "Yes"], p=[0.65, 0.35]))
        streaming_tv.append(np.random.choice(["No", "Yes"], p=[0.5, 0.5]))
        streaming_movies.append(np.random.choice(["No", "Yes"], p=[0.5, 0.5]))

contracts = np.random.choice(["Month-to-month", "One year", "Two year"], size=n_samples, p=[0.55, 0.2, 0.25])
paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])
payment_methods = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    size=n_samples,
    p=[0.35, 0.25, 0.2, 0.2]
)

# Calculate Monthly Charges realistically based on services
monthly_charges = []
for idx in range(n_samples):
    charge = 20.0  # Base charge
    if phone_service[idx] == "Yes":
        charge += 10.0
    if multiple_lines[idx] == "Yes":
        charge += 15.0
    
    if internet_service[idx] == "DSL":
        charge += 30.0
    elif internet_service[idx] == "Fiber optic":
        charge += 50.0
        
    if online_security[idx] == "Yes": charge += 8.0
    if online_backup[idx] == "Yes": charge += 8.0
    if device_protection[idx] == "Yes": charge += 8.0
    if tech_support[idx] == "Yes": charge += 8.0
    if streaming_tv[idx] == "Yes": charge += 12.0
    if streaming_movies[idx] == "Yes": charge += 12.0
    
    # Add minor noise
    charge += np.random.normal(0, 2)
    monthly_charges.append(round(max(18.0, charge), 2))

# Total Charges
total_charges = []
for idx in range(n_samples):
    if tenures[idx] == 0:
        total_charges.append(" ")  # Represent missing/blank values like real dataset
    else:
        tc = tenures[idx] * monthly_charges[idx] + np.random.normal(0, 5)
        total_charges.append(round(max(monthly_charges[idx], tc), 2))

# Realistic Churn Logic
# Base probability of churn
churn_probs = []
for idx in range(n_samples):
    prob = 0.15 # base rate
    
    # Higher for month-to-month contracts
    if contracts[idx] == "Month-to-month":
        prob += 0.35
    elif contracts[idx] == "One year":
        prob += 0.05
        
    # Lower for longer tenure
    if tenures[idx] < 12:
        prob += 0.2
    elif tenures[idx] > 48:
        prob -= 0.1
        
    # Higher for Fiber Optic
    if internet_service[idx] == "Fiber optic":
        prob += 0.15
        
    # Lower if they have security or tech support
    if online_security[idx] == "Yes":
        prob -= 0.1
    if tech_support[idx] == "Yes":
        prob -= 0.1
        
    # Higher charges
    if monthly_charges[idx] > 80:
        prob += 0.1
        
    # Clamp probability between 0.01 and 0.99
    prob = max(0.01, min(0.99, prob))
    churn_probs.append(prob)

churns = [("Yes" if np.random.rand() < prob else "No") for prob in churn_probs]

# Create DataFrame
df = pd.DataFrame({
    "customerID": customer_ids,
    "gender": genders,
    "SeniorCitizen": senior_citizens,
    "Partner": partners,
    "Dependents": dependents,
    "tenure": tenures,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contracts,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_methods,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churns
})

# Save to CSV
df.to_csv("telecom_churn.csv", index=False)
print("telecom_churn.csv successfully generated with 5000 rows.")
