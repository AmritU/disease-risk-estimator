import pandas as pd
from xgboost import XGBClassifier
import joblib

# 1. Load the original dataset
df = pd.read_csv('../data/disease_dataset.csv')

# 2. Manually map text to numbers (Clean and predictable)
# Gender: Male = 1, Female = 0
df['Gender_Male'] = df['Gender'].map({'Male': 1, 'Female': 0})

# Smoking: Any smoking history = 1, Never smoked = 0
df['Smoking_History'] = df['Smoking_Status'].map({'Current': 1, 'Former': 1, 'Never': 0})

# 3. Force the EXACT column sequence your Streamlit app will use
features = ['Glucose', 'BMI', 'Gender_Male', 'Smoking_History']
X = df[features] 
y = df['Diabetes_Diagnosis']

# 4. Train the XGBoost Model directly on raw numbers (No scaler!)
print("Training the XGBoost Model...")
model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X, y)

# 5. Overwrite the old model file
joblib.dump(model, '../models/diabetes_model.pkl')
print("Clean model successfully trained and saved!")