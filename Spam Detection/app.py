import streamlit as st
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')
ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('Vectorizer.pkl', 'rb'))
mnb = pickle.load(open('Model.pkl', 'rb'))

st.title('Spam Detector')

given_text = st.text_input('Enter a message')

if st.button('Predict'):
    text_input = transform_text(given_text)

    vector = tfidf.transform([text_input])  # Wrap text_input in a list

    result = mnb.predict(vector)

    if result[0] == 1:
        st.header('Spam Detected')
    else:
        st.header('Not Spam Detected')
