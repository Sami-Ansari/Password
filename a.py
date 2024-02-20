import streamlit as st
import dill
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
model = dill.load(open("xgb_classifier.pkl", "rb"))
vectorizer=dill.load(open("vectorizer.pkl","rb"))
async def  make_chars(inputs):
    characters=[]
    for letter in inputs:
        characters.append(letter)
    return characters



# Define Streamlit app
st.set_page_config(page_title="Login", page_icon=":lock:")
st.markdown('<style>body{background-color: #0f0f0f; color: #00ff00;}</style>', unsafe_allow_html=True)
st.header("Password Strength Analyzer")

# Create username input box
username = st.text_input("Enter Username")

# Create password input field
password = st.text_input("Enter Password", type="password")

# Show password checkbox
# show_password = st.checkbox("Show Password")

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

# Toggle password visibility
# if show_password:
#     st.text_input("Your Password", value=password)
