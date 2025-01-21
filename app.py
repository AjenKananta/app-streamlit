import streamlit as st
import pandas as pd
import joblib

# Load models
rf_model = joblib.load('random_forest_model.pkl')  # Ganti nama file jika berbeda
svm_model = joblib.load('svm_model.pkl')          # Ganti nama file jika berbeda

# App title
st.title("Credit Risk Classification App")
st.write("Aplikasi ini memprediksi risiko kredit (Good/Bad Risk) menggunakan Random Forest dan SVM.")

# Input data manual
st.sidebar.header("Input Data")
st.sidebar.write("Masukkan data secara manual:")
age = st.sidebar.number_input("Age", min_value=18, max_value=100, step=1)
income = st.sidebar.number_input("Income", min_value=0, step=500)
home_ownership = st.sidebar.selectbox("Home Ownership (0=RENT, 1=OWN, etc.)", options=[0, 1, 2, 3])
emp_length = st.sidebar.number_input("Employment Length (years)", min_value=0, max_value=50, step=1)
loan_intent = st.sidebar.selectbox("Loan Intent (0=Personal, 1=Education, etc.)", options=[0, 1, 2, 3, 4, 5])
loan_grade = st.sidebar.selectbox("Loan Grade (0=A, 1=B, etc.)", options=[0, 1, 2, 3, 4, 5, 6])
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=100)
loan_interest_rate = st.sidebar.number_input("Loan Interest Rate", min_value=0.0, step=0.01)
loan_percent_income = st.sidebar.number_input("Loan Percent Income", min_value=0.0, max_value=1.0, step=0.01)
default_on_file = st.sidebar.selectbox("Default on File (0=No, 1=Yes)", options=[0, 1])

# Buat DataFrame untuk data manual
input_data = pd.DataFrame({
    'person_age': [age],
    'person_income': [income],
    'person_home_ownership': [home_ownership],
    'person_emp_length': [emp_length],
    'loan_intent': [loan_intent],
    'loan_grade': [loan_grade],
    'loan_amnt': [loan_amount],
    'loan_int_rate': [loan_interest_rate],
    'loan_percent_income': [loan_percent_income],
    'cb_person_default_on_file': [default_on_file]
})

st.write("Data manual yang dimasukkan:")
st.write(input_data)

# Pilih model
st.sidebar.header("Pilih Model")
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "SVM"])

# Prediksi
if st.sidebar.button("Predict"):
    model = rf_model if model_choice == "Random Forest" else svm_model
    
    # Lakukan prediksi
    prediction = model.predict(input_data)
    input_data['Prediction'] = ["Good Risk" if p == 1 else "Bad Risk" for p in prediction]
    
    # Tampilkan hasil prediksi
    st.write("Hasil Prediksi:")
    st.write(input_data)

# Catatan tentang model
st.sidebar.info("""
- Random Forest: Model berbasis pohon keputusan dengan ensemble learning.
- SVM (Support Vector Machine): Model berbasis hyperplane untuk klasifikasi.
""")
