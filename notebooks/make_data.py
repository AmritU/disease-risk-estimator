import pandas as pd
import numpy as np

# Generates 100 fake patient records
data = {
    'Glucose': np.random.randint(70, 200, 100),
    'BMI': np.random.uniform(18.5, 40.0, 100),
    'Gender': np.random.choice(['Male', 'Female'], 100),
    'Smoking_Status': np.random.choice(['Never', 'Former', 'Current'], 100),
    'Diabetes_Diagnosis': np.random.choice([0, 1], 100)
}

# Create the DataFrame and save it to the data folder
df = pd.DataFrame(data)
df.to_csv('../data/disease_dataset.csv', index=False)

print("Dummy dataset successfully created in the data folder!")