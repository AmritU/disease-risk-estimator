import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load the Dataset
# Assuming you downloaded a dataset and placed it in the 'data' folder
df = pd.read_csv('../data/disease_dataset.csv')

# 2. Handle Missing Values
# Medical data often has blanks. We fill missing numerical columns with the median 
# so outliers don't skew the data.
df['BMI'] = df['BMI'].fillna(df['BMI'].median())
df['Glucose'] = df['Glucose'].fillna(df['Glucose'].median())

# 3. Encode Categorical Variables
# ML models only understand numbers. We must convert text columns (like Gender or Smoking Status).
# get_dummies converts 'Male'/'Female' into 1s and 0s. 
df = pd.get_dummies(df, columns=['Gender', 'Smoking_Status'], drop_first=True)

# 4. Separate Features (Inputs) and Target (Output)
# 'X' contains the health metrics. 'y' contains the target answer (e.g., 1 for Diabetes, 0 for healthy)
X = df.drop(columns=['Diabetes_Diagnosis']) 
y = df['Diabetes_Diagnosis']

# 5. Train/Test Split
# We hide 20% of the data (test set) to quiz the model later and see if it actually learned.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Feature Scaling (Crucial Step)
# Glucose can be 150, but BMI might be 25. Without scaling, the model thinks Glucose is 
# mathematically "more important" simply because the number is bigger.
scaler = StandardScaler()

# We "fit" (learn the math) and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# We ONLY transform the test data (never fit). This prevents "data leakage" (cheating on the quiz)
X_test_scaled = scaler.transform(X_test)

# 7. Save the Scaler for the Web App
# We MUST save this scaler. Later, when a user types "Glucose = 110" into your Streamlit app, 
# you have to scale their input using this exact same mathematical rule before predicting.
joblib.dump(scaler, '../models/standard_scaler.pkl')

print("Preprocessing complete! Data is ready for training.")