# Customer Churn Prediction Project

This project implements a complete, professional machine learning pipeline to predict customer churn for a telecom company. It features a realistic data generation script, exploratory data analysis (EDA), robust preprocessing pipelines, model training and comparison, and model evaluation metrics.

---

## Project Structure

```text
├── plots/                           # Folder containing generated plots
│   ├── churn_distribution.png       # Class balance of the target variable
│   ├── tenure_vs_churn.png          # Density of tenure for churned vs stayed customers
│   ├── monthly_charges_vs_churn.png # Average monthly charges for churned vs stayed customers
│   ├── best_model_confusion_matrix.png # Evaluation confusion matrix
│   └── feature_importances.png      # Feature importance rankings (Random Forest)
├── generate_data.py                 # Script to generate synthetic telecom churn data
├── telecom_churn.csv                # The generated dataset (5,000 customers)
├── main.py                          # The main machine learning pipeline script
├── README.md                        # Project overview and run instructions (This file)
└── DOCUMENTATION.md                 # Deep-dive explanations of each step
```

---

## Installation & Setup

1. Make sure you have **Python 3.8+** installed.
2. Install the required external libraries using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run

1. **Step 1: Generate the dataset**
   Run the data generator to create the `telecom_churn.csv` file with 5,000 customer records:
   ```bash
   python generate_data.py
   ```

2. **Step 2: Run the Machine Learning Pipeline**
   Run the main script to process the data, save visualizations in the `plots/` folder, train the models, and output predictions:
   ```bash
   python main.py
   ```

---

## Quick Summary of Results

- **Dataset Size:** 5,000 customers, 21 columns (both numeric and categorical).
- **Models Compared:** Logistic Regression vs. Random Forest Classifier.
- **Best Model:** Logistic Regression (approx. **70.2%** test accuracy).
- **Key Findings:** Customers with low tenure, month-to-month contracts, and high monthly charges are at the highest risk of churning.
"# Customer_Churn_prediction" 
