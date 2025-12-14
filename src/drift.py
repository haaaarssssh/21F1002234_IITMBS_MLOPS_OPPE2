import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Load reference (training) and current (generated) data
reference_df = pd.read_csv("data/data.csv").dropna()
current_df = pd.read_csv("data/generated_data.csv")

# Evidently requires column names to match exactly.
# The generated data doesn't have a 'target' column, which is fine.
# We select the common columns for comparison.
common_cols = list(set(reference_df.columns) & set(current_df.columns))
reference_df = reference_df[common_cols]
current_df = current_df[common_cols]


# Create and run the data drift report
drift_report = Report(metrics=[
    DataDriftPreset(),
])

drift_report.run(reference_data=reference_df, current_data=current_df)

# Save the report as an HTML file
drift_report.save_html("data_drift_report.html")

print("\n--- Data Drift Analysis (Task 7) ---")
print("Data drift report has been generated: data_drift_report.html")
print("Open this file to see a detailed comparison of feature distributions.")
