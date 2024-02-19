import streamlit as st
import dill
import numpy as np

# Define Streamlit app
st.set_page_config(page_title="Login", page_icon=":lock:")
st.markdown('<style>body{background-color: #0f0f0f; color: #00ff00;}</style>', unsafe_allow_html=True)
st.header("Password Strength Analyzer")

# Load model and vectorizer
@st.cache_resource()
def load_model_and_vectorizer():
    model = dill.load(open("xgb_classifier.pkl", "rb"))
    vectorizer = dill.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# Create username input box
username = st.text_input("Enter Username")

# Create password input field
password = st.text_input("Enter Password", type="password")

# Password strength prediction
if st.button("Login"):
    x_pred = np.array([password])
    x_pred = vectorizer.transform(x_pred)
    predicted = model.predict(x_pred)
    probab = model.predict_proba(x_pred)
    if predicted == 1:
        st.success("Strong Password But Watch Out")
    elif predicted <= 0:
        st.warning("Already Cracked Somewhere Choose Strong One")
