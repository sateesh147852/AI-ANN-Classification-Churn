import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

#Load trained model

model = tf.keras.models.load_model('model.h5')

#Load the Trained Model scaler, pickle, onehotencoder

model = tf.keras.models.load_model('model.h5')

#Load encoder and scaler
with open('onehot_encoder_geo.pkl','rb') as file:
    one_encoder_geo = pickle.load(file)

with open('label_encoder.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)    

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)


# stremlit app
st.title('Customer Churn Prediction')

#User input
geography = st.selectbox('Geography', one_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_card = st.selectbox("Has credit card", [0, 1])
is_active_member = st.selectbox("Is Active member", [0, 1])

input_data = ({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

#One-Hot encode Geography
geo_encoded = one_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_encoder_geo.get_feature_names_out(['Geography']))

#Combine one-hot encoded columns with input data
input_data = pd.concat([pd.DataFrame(input_data).reset_index(drop=True), geo_encoded_df], axis = 1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)

#Predict Churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(prediction_proba)

if prediction_proba > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")    



