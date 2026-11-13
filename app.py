

import gradio as gr
import joblib

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

def predict_sentiment(review):
    review = tfidf.transform([review])
    prediction = model.predict(review)
    return prediction[0].capitalize()

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Enter Product Review"),
    outputs=gr.Textbox(label="Predicted Sentiment"),
    title="Product Review Sentiment Analysis"
)

demo.launch()
