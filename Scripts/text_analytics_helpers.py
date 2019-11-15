"""
Author: Saurabh Annadate

This script contains all functions that help with text analytics
"""

import logging
import os
import pandas as pd
import yaml
import logging
import datetime
import re

from Scripts.helpers import update_ratings
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger()

def token_to_index(token, dictionary):
    """
    Given a token and a gensim dictionary, return the token index
    if in the dictionary, None otherwise.
    Reserve index 0 for padding.
    """
    if token not in dictionary.token2id:
        return None
    return dictionary.token2id[token] + 1

def texts_to_indices(text, dictionary):
    """
    Given a list of tokens (text) and a gensim dictionary, return a list
    of token ids.
    """
    result = list(map(lambda x: token_to_index(x, dictionary), text))
    return list(filter(None, result))


def corpus_tokenize(df, process_colname, new_colname):
    """
    Tokenizes the given column in the input dataset

    Args:
        df (Pandas Dataframe): Input Dataframe
        process_colname (String): Name of the column to tokenized
        new_colname (String): Name of the new column

    Returns:
        Pandas Dataframe with the additional column added

    """

    df[new_colname] = df[process_colname].apply(word_tokenize)

    return df


def word_tokenize(text):
    """
    Creates tokens from given text
    """
    tokens = re.findall(r'[A-Za-z0-9@-]+', text.lower())
    
    # get rid of empty tokens
    tokens = list(filter(None, tokens))
    
    return tokens


def pre_process_data(df, resample = False, resample_count = None):
    """
    This function pre-processes the text corpus to make it ready for the model building

    Args:
        df (Pandas Dataframe): Raw data
        resample (Boolean): Whether the data needs to be sampled down or not
        resample_count (int): If resample is True, what to resample the data to

    Returns:
        Pandas Dataframe with the processed data
    """

    #Downsampling the data
    if resample == True:
        df = df.sample(n = resample_count, random_state = 12345, replace = False).reset_index(drop=True)
    
    #Updating the ratings
    df = update_ratings(df)
    
    #Dropping empty reviews
    df = corpus_tokenize(df, "Review", "Review_Tokens")
    df["Token_count"] = df["Review_Tokens"].apply(len)
    df = df[df.Token_count > 0]

    #Dropping columns
    df.drop(['Review_head', 'Token_count', 'Review_Tokens'], inplace=True, axis = 1)

    #Making lower case
    df['Review'] = df['Review'].str.lower()

    return df


    









