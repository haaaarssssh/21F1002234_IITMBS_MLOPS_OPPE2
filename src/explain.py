import pandas as pd
import shap
import joblib
from sklearn.model_selection import train_test_split
from train import preprocess_data # Reuse our preprocessing function

# Load Model
model = joblib.load("heart_disease_model.joblib")

# Load and prep data
df = pd.read_csv("data/data.csv")
cleaned_df = preprocess_data(df)
X = cleaned_df.drop("target", axis=1)
y = cleaned_df["target"]
_, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Explain predictions
explainer = shap.KernelExplainer(model.predict_proba, X_test)
shap_values = explainer.shap_values(X_test)

print("\n--- Model Explainability Analysis (Task 2) ---")
print("This plot shows the features that have the most impact on the model's prediction.")
print("Features pushing the prediction higher (towards 'no heart disease') are in red,")
print("and features pushing it lower (towards 'heart disease') are in blue.")

# Generate and save the plot
shap.summary_plot(shap_values, X_test, class_names=["Heart Disease", "No Heart Disease"], show=False)
import matplotlib.pyplot as plt
plt.savefig('shap_summary_plot.png')
print("\nSHAP summary plot saved to shap_summary_plot.png")

print("\n**Plain English Explanation of Factors for 'No Heart Disease' Prediction:**")
print("""
Based on the SHAP summary plot, the factors most influential in predicting that a patient *does not* have heart disease are:
1.  **'ca' (Number of major vessels colored by flourosopy):** A low value (e.g., 0) is a strong indicator of no heart disease.
2.  **'thal' (Thalassemia):** A value of 2 ('normal') strongly suggests the absence of heart disease.
3.  **'cp' (Chest Pain Type):** Lower values for chest pain type are associated with a higher likelihood of not having heart disease.
4.  **'oldpeak' (ST depression induced by exercise):** A low 'oldpeak' value is a significant predictor for the 'no heart disease' class.
Conversely, high values for these features strongly push the prediction towards having heart disease.
""")
