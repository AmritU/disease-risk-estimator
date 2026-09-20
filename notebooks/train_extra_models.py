import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

print("Training Heart Disease Model...")
# 1. Heart Disease (Age, Cholesterol, BP, Heart Rate)
heart_data = pd.DataFrame({
    'Age': np.random.randint(30, 80, 100),
    'Cholesterol': np.random.randint(150, 300, 100),
    'Blood_Pressure': np.random.randint(90, 180, 100),
    'Heart_Rate': np.random.randint(60, 120, 100),
    'Heart_Disease': np.random.choice([0, 1], 100)
})

X_heart = heart_data.drop('Heart_Disease', axis=1)
y_heart = heart_data['Heart_Disease']

heart_scaler = StandardScaler()
X_heart_scaled = heart_scaler.fit_transform(X_heart)

heart_model = RandomForestClassifier(max_depth=5, random_state=42).fit(X_heart_scaled, y_heart)

joblib.dump(heart_model, '../models/heart_model.pkl')
joblib.dump(heart_scaler, '../models/heart_scaler.pkl')

print("Training Kidney Disease Model...")
# 2. Kidney Disease (Blood Pressure, Creatinine, Blood Urea)
kidney_data = pd.DataFrame({
    'Blood_Pressure': np.random.randint(90, 180, 100),
    'Creatinine': np.random.uniform(0.5, 5.0, 100),
    'Blood_Urea': np.random.randint(10, 100, 100),
    'Kidney_Disease': np.random.choice([0, 1], 100)
})

X_kidney = kidney_data.drop('Kidney_Disease', axis=1)
y_kidney = kidney_data['Kidney_Disease']

kidney_scaler = StandardScaler()
X_kidney_scaled = kidney_scaler.fit_transform(X_kidney)

kidney_model = RandomForestClassifier(max_depth=5, random_state=42).fit(X_kidney_scaled, y_kidney)

joblib.dump(kidney_model, '../models/kidney_model.pkl')
joblib.dump(kidney_scaler, '../models/kidney_scaler.pkl')

print("Success! Heart and Kidney models & scalers saved to /models.")