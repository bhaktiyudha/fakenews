import string

import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Read data
truenews = pd.read_csv('news_true.csv')
fakenews = pd.read_csv('news_fake.csv')
truenews['True/Fake'] = 'True'
fakenews['True/Fake'] = 'Fake'

# Combine the 2 DataFrames into a single data frame
news = pd.concat([truenews, fakenews])
news["Article"] = news["title"] + news["text"]
news.sample(frac=1)  # Shuffle 100%

# Data Cleaning
nltk.download('stopwords')


def process_text(s):
    # Check string to see if they are punctuation
    nopunc = [char for char in s if char not in string.punctuation]
    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    # Convert string to lowercase and remove stopwords
    clean_string = [word for word in nopunc.split() if word.lower() not in stopwords.words('indonesian')]
    return clean_string


# Tokenize the text
news['Clean Text'] = news['Article'].apply(process_text)

# Train Naive Bayes Model
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=process_text)),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB())
])

pipeline.fit(news['Clean Text'], news['True/Fake'])


# Streamlit app
st.title("Fake News Detection App")

# Input text box for user input
user_input = st.text_input("Enter the news text:")

if user_input:
    # Preprocess the input text
    cleaned_input = process_text(user_input)
    input_text = ' '.join(cleaned_input)

    # Predict using the trained model
    prediction = pipeline.predict([input_text])[0]

    # Display the result
    st.subheader("Prediction:")
    st.write(f"The input news is classified as: {prediction}")
