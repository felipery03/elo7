import pandas as pd
import pickle
import unidecode
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

nltk.download('rslp', quiet=True)

def normalize_text(string):
    string = unidecode.unidecode(string)
    string = re.sub(r'[^a-zA-Z]', ' ', string.lower())

    return string

    

def tokenize(text):
    ''' Prep a text data casting to lowercase,
        remove punctuation, tokenizing,
        removing stop_words and lemmatinzing.
    Params:
    text (string): Input text data
    Returns:
    result (list): Processed tokens list 
    '''
   

    stop_words = stopwords.words("portuguese")
    stemmer = RSLPStemmer()

    result = normalize_text(text)
    result = word_tokenize(result, language='portuguese')
    
    custom_stop_words = ['x', 'c', 'cm', 's'] 
    stop_words = stop_words + custom_stop_words
    
    result = [stemmer.stem(w) for w in result if w not in stop_words]
    
    return result

def save_model(model, model_filepath):
    ''' Save model fitted in a pickle file.
    Params:
    model (model): Model fitted.
    model_filepath (string): Path to save pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))