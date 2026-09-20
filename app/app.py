import streamlit as st
import joblib
import pandas as pd
import time
from pymongo import MongoClient
import datetime
import certifi

st.set_page_config(page_title="Multi-Disease Risk Dashboard", page_icon="🩺", layout="wide")

# --- 1. DATABASE CONNECTION ---
@st.cache_resource
def init_connection():
    client = MongoClient(st.secrets["MONGO_URI"], tlsCAFile=certifi.where())
    return client

client = init_connection()
db = client.healthcare_app
predictions_collection = db.predictions

# --- 2. LOAD ALL MODELS ---
@st.cache_resource
def load_models():
    # Diabetes
    d_model = joblib.load('../models/diabetes_model.pkl')
    d_scaler = joblib.load('../models/standard_scaler.pkl')
    
    # Heart
    h_model = joblib.load('../models/heart_model.pkl')
    h_scaler = joblib.load('../models/heart_scaler.pkl')
    
    # Kidney
    k_model = joblib.load('../models/kidney_model.pkl')
    k_scaler = joblib.load('../models/kidney_scaler.pkl')
    
    return d_model, d_scaler, h_model, h_scaler, k_model, k_scaler

d_model, d_scaler, h_model, h_scaler, k_model, k_scaler = load_models()

# --- 3. UI LAYOUT ---
st.title("🩺 Multi-Disease Risk Dashboard")
st.divider()

# Using tabs to organize the large number of inputs cleanly
tab1, tab2, tab3 = st.tabs(["Biometrics (All)", "Heart & Blood Vitals", "Kidney Panel"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=120, value=45)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    with col2:
        gender = st.selectbox("Biological Sex", ["Male", "Female"])
        smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])

with tab2:
    col3, col4 = st.columns(2)
    with col3:
        glucose = st.number_input("Fasting Glucose (mg/dL)", value=100.0)
        blood_pressure = st.number_input("Systolic Blood Pressure (mmHg)", value=120.0)
    with col4:
        cholesterol = st.number_input("Cholesterol (mg/dL)", value=200.0)
        heart_rate = st.number_input("Resting Heart Rate (bpm)", value=70.0)

with tab3:
    col5, col6 = st.columns(2)
    with col5:
        creatinine = st.number_input("Creatinine (mg/dL)", value=1.0, step=0.1)
    with col6:
        blood_urea = st.number_input("Blood Urea (mg/dL)", value=30.0)

st.divider()

# --- 4. INFERENCE & RESULTS ---
if st.button("Generate Comprehensive Risk Report", type="primary", use_container_width=True):
    with st.spinner("Running Multi-Model Inference..."):
        time.sleep(0.5)
        
        # 1. Format data arrays exactly as the separate models expect them
        d_data = pd.DataFrame([{
            'Glucose': glucose, 'BMI': bmi, 
            'Gender_Male': 1 if gender == "Male" else 0,
            'Smoking_Status_Former': 1 if smoking == "Former" else 0,
            'Smoking_Status_Never': 1 if smoking == "Never" else 0
        }])
        
        h_data = pd.DataFrame([{
            'Age': age, 'Cholesterol': cholesterol, 
            'Blood_Pressure': blood_pressure, 'Heart_Rate': heart_rate
        }])
        
        k_data = pd.DataFrame([{
            'Blood_Pressure': blood_pressure, 'Creatinine': creatinine, 'Blood_Urea': blood_urea
        }])
        
        # 2. Predict for each model
        d_risk = float(round(d_model.predict_proba(d_scaler.transform(d_data))[0][1] * 100, 2))
        h_risk = float(round(h_model.predict_proba(h_scaler.transform(h_data))[0][1] * 100, 2))
        k_risk = float(round(k_model.predict_proba(k_scaler.transform(k_data))[0][1] * 100, 2))
        
        # 3. Display Results in a 3-column Grid
        res1, res2, res3 = st.columns(3)
        res1.metric("Diabetes Risk", f"{d_risk:.2f}%")
        res2.metric("Heart Disease Risk", f"{h_risk:.2f}%")
        res3.metric("Kidney Disease Risk", f"{k_risk:.2f}%")
        
        # 4. Save Combined Data to MongoDB
        db_document = {
            "timestamp": datetime.datetime.now(datetime.UTC),
            "diabetes_risk": d_risk,
            "heart_risk": h_risk,
            "kidney_risk": k_risk,
            "vitals": {
                "age": float(age), "bmi": float(bmi), "glucose": float(glucose),
                "blood_pressure": float(blood_pressure), "cholesterol": float(cholesterol),
                "heart_rate": float(heart_rate), "creatinine": float(creatinine)
            }
        }
        
        try:
            predictions_collection.insert_one(db_document)
            st.toast("✅ Comprehensive panel saved to database")
        except Exception as e:
            st.error("Database connection failed.")
            print(e)

# --- 5. RETRIEVE & DISPLAY PATIENT HISTORY ---
st.divider()
st.subheader("📋 Recent Patient Assessments")

with st.expander("View Past Predictions", expanded=True):
    try:
        cursor = predictions_collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(10)
        records = list(cursor)
        
        if len(records) == 0:
            st.info("No past predictions found in the database.")
        else:
            flat_records = []
            for r in records:
                vitals = r.get("vitals", r.get("patient_vitals", {}))
                
                flat_records.append({
                    "Date & Time (UTC)": r["timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "Diabetes Risk (%)": r.get("diabetes_risk", r.get("risk_score", 0)),
                    "Heart Risk (%)": r.get("heart_risk", 0),
                    "Kidney Risk (%)": r.get("kidney_risk", 0),
                    
                    # --- THE FIX ---
                    # Replaced "N/A" with None so PyArrow can handle missing numbers
                    "Age": vitals.get("age", None),
                    "BMI": vitals.get("bmi", None),
                    "BP": vitals.get("blood_pressure", None),
                    "Glucose": vitals.get("glucose", None)
                })
            
            # Convert to Pandas DataFrame and render
            df_history = pd.DataFrame(flat_records)
            st.dataframe(df_history, use_container_width=True, hide_index=True)

            # --- 6. MULTI-MODEL TREND VISUALIZATION ---
            st.divider()
            st.markdown("**📉 Comprehensive Risk Trend Analysis**")
            
            # Select the date and all three risk columns
            chart_data = df_history[[
                "Date & Time (UTC)", 
                "Diabetes Risk (%)", 
                "Heart Risk (%)", 
                "Kidney Risk (%)"
            ]].copy()
            
            chart_data = chart_data.sort_values("Date & Time (UTC)")
            chart_data.set_index("Date & Time (UTC)", inplace=True)
            
            # Streamlit automatically plots a multi-line chart when given multiple columns!
            st.line_chart(chart_data)
            
    except Exception as e:
        st.error("Could not fetch history from the database.")
        print(e)