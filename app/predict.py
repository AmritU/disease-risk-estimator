import joblib
import pandas as pd

# 1. Load the "brain" (model) and the "translator" (scaler)
model = joblib.load('../models/diabetes_model.pkl')
scaler = joblib.load('../models/standard_scaler.pkl')

def get_risk_score(patient_data):
    """
    Takes raw patient data, routes it through the scaling process, 
    and returns the AI's risk prediction.
    """
    # Convert the incoming dictionary into a Pandas DataFrame
    df = pd.DataFrame([patient_data])
    
    # Scale the data using the exact mathematical rules learned during training
    scaled_data = scaler.transform(df)
    
    # Get the probability of the positive class (disease)
    # predict_proba returns an array like [[probability_healthy, probability_sick]]
    probability = model.predict_proba(scaled_data)[0][1]
    
    return round(probability * 100, 2)

# --- Testing the Data Flow ---
if __name__ == "__main__":
    # Simulating data flowing in from the future web frontend
    sample_patient = {
        'Glucose': 150.0,
        'BMI': 32.5,
        'Gender_Male': 1,             # 1 for Male, 0 for Female
        'Smoking_Status_Former': 0,   
        'Smoking_Status_Never': 1     # 1 for Never smoked
    }
    
    risk = get_risk_score(sample_patient)
    print(f"System Test Successful! Patient Disease Risk: {risk}%")