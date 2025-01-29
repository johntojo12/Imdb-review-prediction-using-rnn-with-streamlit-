import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#load the dataset word index
word_index=imdb.get_word_index()
reverse_word_inex={value:key for key,value in word_index.items()}

#load model 
model=load_model('best_model_1.h5')

#decode review 
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])

#fn to prepreprocess the imput txt
def preprocessed_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2) for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

# streamlit
import streamlit as st 
st.title("IMDB movie review sentiment analysis")
st.write("Enter the movies review")

#user input 
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input=preprocessed_text(user_input)
    
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.7 else 'Negative'
    
        # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')