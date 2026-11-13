
import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Application title
st.title("Product Review Sentiment Analysis")
st.write("Enter a product review to predict its sentiment.")

# User input
review = st.text_area("Enter your product review:")

# Prediction
if st.button("Predict Sentiment"):
    if review.strip():
        review_tfidf = tfidf.transform([review])
        prediction = model.predict(review_tfidf)

        st.subheader("Predicted Sentiment:")
        st.success(prediction[0].capitalize())
    else:
        st.warning("Please enter a review.")
