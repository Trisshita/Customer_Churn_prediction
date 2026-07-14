# Deep-Dive Project Documentation: Customer Churn Prediction

This document provides a deep, easy-to-understand explanation of each step in this machine learning project. It is structured to help you understand the core concepts of the machine learning pipeline, making it useful for internship reports or presentations.

---

> ## 📌 **PROBLEM STATEMENT**
> Telecom companies face high customer acquisition costs. Losing existing customers (churning) directly impacts revenue. The business needs a way to identify high-risk customers before they cancel their subscriptions so that retention teams can target them with proactive offers.

> ## 💡 **OUR SOLUTION**
> We built an end-to-end Machine Learning pipeline that:
> 1. Simulates realistic customer behavior (5,000 users) with various risk factors (contract type, charges, tech support, etc.).
> 2. Cleans and preprocesses data (scaling numerical values, encoding categorical texts).
> 3. Automatically trains and compares multiple models (Logistic Regression & Random Forest).
> 4. Validates performance using precision, recall, and a confusion matrix to predict high-risk churners (achieving ~70.2% accuracy) and outputting actionable probabilities.

---

## 1. Introduction: What is Customer Churn?

**Customer Churn** (or customer attrition) occurs when customers stop doing business with a company. For a telecom provider, this means a customer canceling their subscription or service.
*   **Why is it important?** Acquiring new customers is much more expensive than retaining existing ones. By predicting which customers are likely to leave, the business can take proactive measures (like offering discounts, promotions, or better support) to keep them.
*   **The Machine Learning Goal:** Build a classification model that takes a customer's profile (e.g., tenure, services subscribed to, billing preferences, charges) and predicts whether they will churn (`Yes` / `1`) or stay (`No` / `0`).

---

## 2. Step 1: Data Generation (`generate_data.py`)

Real-world customer data is complex and messy. To mimic this, our data generation script creates a synthetic dataset of **5,000 customers** with 21 attributes.

We built logical dependencies into the data to simulate real-world behaviors:
1.  **Contract types:** Month-to-month contracts have a much higher base probability of churning compared to 1-year or 2-year contracts.
2.  **Tenure (Loyalty):** New customers (low tenure) are far more likely to leave than long-term customers.
3.  **Services:** Customers using Fiber optic internet or missing Tech Support / Online Security have higher rates of churn.
4.  **Data Quirks:** We generated blank spaces (`" "`) in the `TotalCharges` column for brand-new customers (where `tenure = 0`) because this is a common issue in real telecom datasets (e.g., the Kaggle Telco Churn dataset).

---

## 3. Step 2: Data Loading & Cleaning (`main.py`)

Before feeding data into a model, we must clean it. In python, we use `pandas` to load the dataset and perform cleaning:

```python
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data["TotalCharges"] = data["TotalCharges"].fillna(0.0)
```
*   **Why coerce `TotalCharges`?** Brand-new customers have empty spaces for `TotalCharges` in the CSV, causing Pandas to read the entire column as text (strings). `pd.to_numeric(..., errors="coerce")` converts these strings into float numbers and turns spaces into `NaN` (Not a Number).
*   **Why fill NaNs with `0.0`?** If a customer has a tenure of 0 months, they haven't been billed yet, so their total charges are logically `0.0`.
*   **Mapping Target Variable:** The target variable `Churn` initially contains `"Yes"` and `"No"`. Machine learning algorithms only understand numbers, so we map them: `"Yes" -> 1` and `"No" -> 0`.
*   **Dropping Unnecessary Columns:** We drop `customerID` because random identifiers carry no predictive power and would only confuse the model.

---

## 4. Step 3: Exploratory Data Analysis (EDA)

EDA is the process of visualizing and exploring data to identify patterns, anomalies, and correlations.
*   **Churn Distribution (`plots/churn_distribution.png`):** Shows the balance of the target class. In our data, around 43% of customers have churned. Knowing this helps us understand if the dataset is highly imbalanced.
*   **Tenure vs. Churn (`plots/tenure_vs_churn.png`):** Uses Kernel Density Estimation (KDE) to show the distribution of tenure for churned vs. active customers. It proves that churn peaks heavily in the first few months of subscription.
*   **Monthly Charges vs. Churn (`plots/monthly_charges_vs_churn.png`):** Uses a boxplot to compare monthly bill sizes. Churned customers generally have a higher median monthly bill.

---

## 5. Step 4: Preprocessing Pipeline

This is one of the most critical steps in machine learning. Raw features cannot be fed directly into most models without preprocessing. We separate our features into **Numerical** and **Categorical** columns and apply different rules to each:

### A. Numerical Columns (`tenure`, `MonthlyCharges`, `TotalCharges`)
*   **Method Used:** `StandardScaler` (Standardization).
*   **Why?** Standard scaler shifts the values so that the mean of the column is `0` and the standard deviation is `1`. This is important because features have different scales (e.g., tenure is 0-72, but total charges go up to 8000+). Without scaling, models like Logistic Regression would assume `TotalCharges` is much more important simply because the numbers are larger.

### B. Categorical Columns (e.g., `Contract`, `InternetService`, `PaymentMethod`)
*   **Method Used:** `OneHotEncoder(drop='first')`.
*   **Why?** Machine learning models cannot process text categories like `"DSL"` or `"Fiber optic"`. One-hot encoding creates separate binary columns for each category (e.g., `InternetService_Fiber optic` = 1 or 0). 
*   **Why `drop='first'`?** If a feature has two categories like Gender (Male/Female), we only need one column (`gender_Male` where 1 = Male, 0 = Female) to represent it. Keeping both would create redundant, highly-correlated features, which hurts models like Logistic Regression (a problem known as the dummy variable trap).

### C. ColumnTransformer
We combine the transformations into a single `ColumnTransformer`. This encapsulates all preprocessing steps so they can be applied consistently to both training and test sets, preventing **data leakage**.

---

## 6. Step 5: Model Training & Comparison

We divide our preprocessed data into **Training (80%)** and **Testing (20%)** subsets. We use **stratified splitting** (`stratify=y`) to ensure that the train and test sets have the exact same percentage of churned customers.

We then train and compare two different algorithms:

### A. Logistic Regression
*   **How it works:** It estimates the probability that an instance belongs to a certain class (churn vs. stay) using a logistic sigmoid function.
*   **Pros:** Extremely fast to train, easy to interpret, and acts as an excellent baseline.
*   **Result:** Achieved **~70.2%** test accuracy.

### B. Random Forest Classifier
*   **How it works:** An ensemble learning method that builds multiple Decision Trees during training and merges their predictions together (voting) to get a more accurate and stable prediction.
*   **Pros:** Excellent at capturing non-linear relationships and interactions between features.
*   **Result:** Achieved **~69.1%** test accuracy.

---

## 7. Step 6: Model Evaluation

To evaluate performance, we look beyond simple accuracy:

### A. Classification Report
Accuracy can be misleading if data is imbalanced. We analyze:
*   **Precision:** Out of all customers the model predicted would churn, how many actually did? (Minimizes false alarms).
*   **Recall (Sensitivity):** Out of all customers who actually churned, how many did the model successfully identify? (Minimizes missed opportunities).
*   **F1-Score:** The harmonic mean of precision and recall.

### B. Confusion Matrix (`plots/best_model_confusion_matrix.png`)
A table layout showing predictions vs. actual truths:
*   **True Negative (TN):** Model correctly predicted the customer would STAY.
*   **True Positive (TP):** Model correctly predicted the customer would CHURN.
*   **False Positive (FP - Type I Error):** Model predicted CHURN, but customer actually STAYED.
*   **False Negative (FN - Type II Error):** Model predicted STAY, but customer actually CHURNED.

### C. Feature Importances (`plots/feature_importances.png`)
Using the Random Forest model, we can extract which features were most influential. The results show that **tenure**, **contract type**, and **monthly charges** play the largest roles in predicting customer churn, which aligns with real-world industry trends.

---

## 8. Step 7: Predicting on New Data

Once the pipeline is trained, the best model can make predictions on unseen, raw data. We defined a new hypothetical customer profile (e.g., month-to-month contract, low tenure, high charges) and passed it to our pipeline. 

The pipeline preprocessed the new data using the same scaler/encoder trained on the history dataset and predicted:
*   **Prediction:** The customer will **CHURN**
*   **Probability:** **85.5%** probability of churning.

This demonstrates how a business can use this model in production: they can score active customer profiles daily and flag those with high churn probabilities for retention campaigns.
