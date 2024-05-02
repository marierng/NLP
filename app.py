import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
import streamlit as st


def preprocessing_helper_function(sms):
  words = word_tokenize(sms) # tokenization
  words = [word.lower() for word in words if word.isalnum()] # Lowercasing
  words = [word for word in words if word not in stopwords.words('english')] # Removing Stopwords
  return " ".join(words)


model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')


def main():
  st.title("Marie's First App!!!!!!!!!!")
  text = st.text_area("Enter a sentence here:")
  if st.button('Predict'):
    preprocessed_text = preprocessing_helper_function(text)
    transformed_text = tfidf.transform([preprocessed_text])
    prediction = model.predict(transformed_text)
    result = "Spam" if prediction[0] == 1 else "Not Spam"
    st.write(f"Prediction: {result}")

if __name__ == '__main__':
  main()    