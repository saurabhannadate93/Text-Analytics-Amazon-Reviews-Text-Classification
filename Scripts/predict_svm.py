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

from Scripts.text_analytics_helpers import corpus_tokenize, texts_to_indices, token_to_index, word_tokenize

logger = logging.getLogger()

def predict_cnn_case(svm_model, text):
    """
    Predict test label using trained svm model for an individual case

    Args:
        svm_model: SVM Model Object
        text: Actual Text
    
    Returns:
        Dictionary with two keys:
            label: Actual predicted label
            probability: Probability of prediction
	"""
    
    logger.debug("Running the predict_svm_case function now.")

    y_pred = svm_model.predict([text])
    
    res_dict = {}
    res_dict["Text"] = text
    res_dict['label'] = str(y_pred[0])
    
    return res_dict


def predict_svm():
    """
    Predict test label using trained svm model

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
    model = pickle.load(open(os.path.join(config["models"]["save_location"], "SVM.pkl"), 'rb'))
    
    #Read txt as a list of string
    with open(os.path.join("Tests","text.txt")) as myfile:
        review = " ".join(line.rstrip() for line in myfile)

    res_dict = predict_cnn_case(model, review)

    #Saving results
    with open(os.path.join("Outputs",'svm_result.json'), 'w') as fp:
        json.dump(res_dict, fp)

    return
