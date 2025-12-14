import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
from train import preprocess_data

# Load Model and Data
model = joblib.load("heart_disease_model.joblib")
df = pd.read_csv("data/data.csv")
cleaned_df = preprocess_data(df)
X = cleaned_df.drop("target", axis=1)
y_true = cleaned_df["target"]
_, X_test, _, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

# Make predictions
y_pred = model.predict(X_test)

# --- Fairness Analysis ---
# Bucket age into 20-year bins as requested
age_bins = [0, 20, 40, 60, 80, 100]
X_test['age_group'] = pd.cut(X_test['age'], bins=age_bins, labels=['0-20', '21-40', '41-60', '61-80', '81-100'])

# Calculate metrics using fairlearn
metrics = {
    'accuracy': (lambda y_true, y_pred: (y_true == y_pred).mean()),
    'selection_rate': (lambda y_true, y_pred: y_pred.mean())
}
grouped_on_age = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=X_test['age_group'])

print("\n--- Fairness Analysis (Task 3) ---")
print("\nMetrics grouped by age group:")
print(grouped_on_age.by_group)

# Calculate demographic parity difference
dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=X_test['age_group'])
print(f"\nDemographic Parity Difference: {dpd:.4f}")
print("""
**Explanation:** Demographic Parity requires that the selection rate (the proportion of individuals predicted to have heart disease)
is the same across all age groups. A value of 0 is perfectly fair. Our value indicates a disparity in prediction rates between the
age groups with the highest and lowest selection rates.
""")
