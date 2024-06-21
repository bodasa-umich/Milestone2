# -*- coding: utf-8 -*-
"""
Code that runs predict (under the main function)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
import pickle

import spacy
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
STOP_WORDS.add('Enron') #add the word enron
STOP_WORDS.add('enron')
STOP_WORDS.add('ENRON')

nlp = spacy.load("en_core_web_sm")

def load_models(vectorizer_source_path='models/tfidf_vectorizer.pkl', model_source_path='models/logistic_regression_model.pkl'):

  #Load TF-IDF vectorizer
  with open( vectorizer_source_path, 'rb') as f:
      vectorizer_augmented_final = pickle.load(f)

  #Load Logistic Regression model
  with open(model_source_path, 'rb') as f:
      best_model_final = pickle.load(f)

  return vectorizer_augmented_final, best_model_final

def calculate_word_lengths(text_data):
    '''
    Function to calculate word lengths
    '''
    return text_data.apply(lambda x: len(x.split())).values.reshape(-1, 1)

def preprocess_text(df, column='content'):
    '''
    Function that converts all text to lower case, eliminates stop words and punctuations and returns the cleaned text
    '''

    #df[column] = df[column].astype(str).str.lower().str.replace('[^\w\s]', ' ', regex=True).str.replace('\n', ' ', regex=True)
    df[column] = df[column].astype(str).str.replace('[^\w\s]', ' ', regex=True).str.replace('\n', ' ', regex=True)


    def clean_text(text):
        doc = nlp(text)
        tokens = []
        prev_token=None
        for token in doc:
            if token.ent_type_ == 'PERSON':
              if prev_token!='person':
                tokens.append('person')
                prev_token= tokens[-1]
            elif token.text not in STOP_WORDS and not token.is_punct and not token.is_space:
                tokens.append(str(token.lemma_).lower())
                prev_token= tokens[-1]
        return ' '.join(tokens)

    df[column] = df[column].apply(clean_text)

    return df

def preprocess_and_predict(x_data, vectorizer, model):
    '''
    Function to preprocess text data, vectorize it using a TF-IDF vectorizer, add word length as a feature,
    stack it with the TF-IDF vectors, and make predictions using the provided model.

    Parameters:
    - y_data: pandas Series containing text data.
    - vectorizer: fitted TF-IDF vectorizer.
    - model: trained LR model.

    Returns:
    - predictions: List of predicted labels.
    '''
    # Preprocess test data
    x_data2= pd.DataFrame({'content': x_data})
    x_data2= preprocess_text(x_data2, 'content')
    x_data= x_data2['content']

    # Vectorize
    X_data_vectorized = vectorizer.transform(x_data)

    # Calculate word lengths
    word_lengths = calculate_word_lengths(x_data)

    # Stack TF-IDF vectors and word lengths
    X_data_combined = hstack((X_data_vectorized, word_lengths)).tocsr()

    # Predict using the model
    predictions = model.predict(X_data_combined)

    return predictions

def predict(csv_path, column_name, vectorizer, model):
  df= pd.read_csv(csv_path)
  predictions = preprocess_and_predict(df[column_name], vectorizer, model)
  df['Prediction']= predictions
  df.to_csv(csv_path, index=False)
  return True

if __name__ == '__main__':
    '''
    You can pass the filename here as a string, and it returns an updated csv with the predictions in column 'Prediction'.
    '''
    vectorizer, model = load_models('models/tfidf_vectorizer.pkl', 'models/logistic_regression_model.pkl')
    success= predict('data/sample_emails_to_predict.csv - Sheet1.csv', 'Email', vectorizer, model)
    print (success)