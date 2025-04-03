import streamlit as st
from transformers import pipeline

# get model checkpoint for the prediction
model_predict = pipeline("text-classification", model="./models/toxicity-classifier")


def predict(text):
    """Predict the label given some text.

    Return the label (as a string).
    """
    output = model_predict(text)

    # Example output [{'label': 'LABEL_0', 'score': 0.9361400008201599}]
    # LABEL_0 -> non-toxic
    # LABEL_1 -> toxic

    return "non-toxic" if output[0]["label"] == "LABEL_0" else "toxic"


# Streamlit UI
st.title("News Headline Toxicity Classifier")
st.write("Enter a news headline to check if it's toxic or not.")

headline = st.text_input("News Headline:")

if st.button("Predict"):
    if headline:
        result = predict(headline)
        st.write(f"Prediction: **{result}**")
    else:
        st.write("Please enter a headline.")
