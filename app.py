import pickle
import streamlit as st
import pandas as pd
import nltk
import numpy as np
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

voting_model = pickle.load(open("votingclassifier.pkl",'rb'))
naiveBayes_model = pickle.load(open("multinoimalnaiveBayes.pkl",'rb'))
vectorizer = pickle.load(open("vectorizer.pkl",'rb'))
def Preprocess_txt(text):
    # lowercase
    text = text.lower()
    # Tokenization
    text = nltk.word_tokenize(text)
    # Special Chracters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # StopWords and Punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)


st.title('Email-Spam Classifier')

sms = st.text_area("Enter Email: ")



if st.button("Predict"):
    # predprocess
    transform_sms = Preprocess_txt(sms)
    # vector
    vector_sms = vectorizer.transform([transform_sms]).toarray()
    # predict
    result = voting_model.predict(vector_sms)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

    