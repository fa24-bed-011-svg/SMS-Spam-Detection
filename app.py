
import streamlit as st
import joblib

model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("SMS Spam Detection System")

st.write(
    "Enter a message to check whether it is Spam or Ham."
)

message = st.text_area("Enter your SMS message:")

if st.button("Predict"):

    if message.strip():

        message = message.lower()

        features = tfidf.transform([message])

        prediction = model.predict(features)[0]

        if prediction == "spam":
            st.error("This message is SPAM!")

        else:
            st.success("This message is HAM (Not Spam)!")

    else:
        st.warning("Please enter a message.")
