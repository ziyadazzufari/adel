import pickle
import streamlit as st 
import tensorflow as tf

# Load the knn model from file
model = tf.keras.models.load_model('ann.h5')
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Mapping
ts_mapping = {
    'Academic' : 0,
    'Vocational' : 1
}

sa_mapping = {
    'A' : 0,
    'B' : 1
}

gender_mapping ={
    'Female' : 0,
    'Male' : 1
}

interest_mapping ={
    'Interested' : 0,
    'Less Interested' : 1,
    'Not Interested' : 2,
    'Uncertain' : 3,
    'Very Interested' : 4
}

res_mapping = {
    'Rural' : 0,
    'Urban' : 1
}

pwic_mapping ={
    'False' : 0,
    'True' : 1
}

def main():
    st.title('Predictive Model Using ANN-Classification for Student Want To Go To College')

    # Input fields
    type_school = st.selectbox('TYPE SCHOOL', list(ts_mapping.keys()))
    school_accreditation = st.selectbox('SCHOOL ACCREDITATION', list(sa_mapping.keys()))
    gender = st.selectbox('GENDER', list(gender_mapping.keys()))
    interest = st.selectbox('INTEREST', list(interest_mapping.keys()))
    residence = st.selectbox('RESIDENCE', list(res_mapping.keys()))
    parent_salary = st.number_input("Parent Salary", value=0)
    house_area = st.number_input("House Area", value=0)
    parent_age = st.number_input("Parent Age", value=0)
    average_grades = st.number_input("Average Grade", value=0)
    parent_was_in_college = st.selectbox('PARENT WAS IN COLLEGE', list(pwic_mapping.keys()))

    # Preprocess
    preprocessed_data = scaler.transform([[parent_age, parent_salary, house_area, average_grades]])

    # Make prediction
    if st.button("Predict"):
        prediction = predict(preprocessed_data, type_school, school_accreditation, gender, interest, residence, parent_was_in_college)
        st.write("Prediction:", prediction)

import numpy as np

def predict(preprocessed_data, type_school, school_accreditation, gender, interest, residence, parent_was_in_college):
    typeschool_numeric = ts_mapping.get(type_school, -1)
    sa_numeric = sa_mapping.get(school_accreditation, -1)
    gender_numeric = gender_mapping.get(gender, -1)
    interest_numeric = interest_mapping.get(interest, -1)
    res_numeric = res_mapping.get(residence, -1)
    pwic_numeric = pwic_mapping.get(parent_was_in_college, -1)
    if typeschool_numeric == -1 or sa_numeric == -1 or gender_numeric == -1 or interest_numeric == -1 or res_numeric == -1 or pwic_numeric == -1:
        return "Invalid Prediction Result"
    
    # Prepare the input data for prediction
    prediction_input = [[preprocessed_data[0][0], preprocessed_data[0][1], preprocessed_data[0][2], preprocessed_data[0][3], typeschool_numeric, sa_numeric, gender_numeric, interest_numeric, res_numeric, pwic_numeric]]
    
    # Perform prediction
    prediction = model.predict(prediction_input)
    if np.allclose(prediction[0], 0):
        return 'Siswa Tidak Melanjutkan Kuliah'
    else:
        return 'Siswa Akan Melanjutkan Kuliah'


# Run the app
if __name__ == '__main__':
    main()
