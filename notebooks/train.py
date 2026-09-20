import pandas as pd
from xgboost import XGBClassifier
import joblib

# 1. Load the preprocessed dummy data
df = pd.read_csv('../data/disease_dataset.csv')
df = pd.get_dummies(df, columns=['Gender', 'Smoking_Status'], drop_first=True)

X = df.drop(columns=['Diabetes_Diagnosis']) 
y = df['Diabetes_Diagnosis']

# Load the scaler we made earlier to transform the data
scaler = joblib.load('../models/standard_scaler.pkl')
X_scaled = scaler.transform(X)

# 2. Train the Model
print("Training the XGBoost Model...")
model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_scaled, y)

# 3. Save the Model
joblib.dump(model, '../models/diabetes_model.pkl')
print("Model successfully trained and saved to the /models directory!")