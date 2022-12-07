import pickle

import streamlit as st
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf


@st.cache(allow_output_mutation=True)
def load():
    k_model = tf.keras.models.load_model('./streamlit-deploy/models/training_cleaned.h5')
    tokenizer_model = pickle.load(open("./streamlit-deploy/models/tokenizer.pkl", 'rb'))
    # return the deep copy of the model
    deep_copy = tf.keras.models.clone_model(k_model)
    deep_copy.set_weights(k_model.get_weights())
    return deep_copy, tokenizer_model


def decode_sentiment(score, include_neutral=True):
    if include_neutral:
        label = "Neutral"
        if score <= 0.5:
            label = "Negative"
        elif score >= 0.5:
            label = "Positive"

        return label
    else:
        return "Negative" if score < 0.5 else "Positive"


model, tokenizer = load()


def predict(_text, include_neutral=True):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([_text]), maxlen=100)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score)}


st.title("Twitter Sentiment Analysis")

text = st.text_input("Enter your text here")
submit = st.button("Submit")

if submit & (text != ""):
    st.write("You entered: ", text)
    result = predict(text)
    st.text(f"Sentiment: {result['label']} \nConfidence: {result['score']}")
