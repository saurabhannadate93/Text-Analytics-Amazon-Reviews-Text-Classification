"""
Author: Saurabh Annadate

This script trains svm models to classify the text into categories
"""

import os
import yaml
import logging
import pickle

import json
import pandas as pd
import numpy as np
import string
from gensim.corpora import Dictionary
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from Scripts.text_analytics_helpers import corpus_tokenize, texts_to_indices, token_to_index, word_tokenize

logger = logging.getLogger()

def predict_cnn_case(cnn_model, X_test_data, text):
    """
    Predict test label using trained cnn model for an individual case

    Args:
        cnn_model: Conv Net Model Object
        X_test_data: Input data tokenized and indicized
        text: Actual Text
    
    Returns:
        Dictionary with two keys:
            label: Actual predicted label
            probability: Probability of prediction
	"""
    
    logger.debug("Running the predict_cnn_case function now.")

    y_pred_prob = cnn_model.predict(X_test_data)
    y_pred = cnn_model.predict_classes(X_test_data)
    
    res_dict = {}
    res_dict["Text"] = text
    res_dict['label'] = str(y_pred[0])
    res_dict['probability'] = str(max(y_pred_prob[0]))
    
    return res_dict


def predict_cnn():
    """
    Predict test label using trained cnn model

    Args:
       None
    
    Returns:
        None
	"""
    logger.debug("Running the predict_cnn function now.")

    #Loading the configuration
    with open(os.path.join("config","config.yml"), "r") as f:
        config = yaml.safe_load(f)
    
    #Loading the model
    model = load_model(os.path.join(config["models"]["save_location"],'amazon_cnn.model'))

    #Read txt as a list of string
    with open(os.path.join("Tests","text.txt")) as myfile:
        review = " ".join(line.rstrip() for line in myfile)

    #Reading MSL
    MAX_SEQUENCE_LENGTH = pickle.load(open(os.path.join(config["models"]["save_location"],'cnn_MSL'), 'rb'))

    #Tokenizing
    review_tok = word_tokenize(review)

    #Loading gensim dictionary
    mydict = Dictionary.load(os.path.join(config["summary_stats"]["save_location"], "amazon.dict"))

    #Convert all text to Indices and padding
    review_ind = texts_to_indices(review_tok, mydict)
    review_ind = pad_sequences([review_ind], maxlen=MAX_SEQUENCE_LENGTH)
    
    #Predicting results
    pred_res = predict_cnn_case(model, review_ind, review)

    #Saving results
    with open(os.path.join("Outputs",'cnn_result.json'), 'w') as fp:
        json.dump(pred_res, fp)

    return
