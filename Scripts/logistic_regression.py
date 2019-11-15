"""
Author: Saurabh Annadate

This script trains logistic regression models to classify the text into categories
"""

import logging
import os
import pandas as pd
import yaml
import logging
import datetime
import pickle

from Scripts.text_analytics_helpers import pre_process_data
from Scripts.helpers import load_data

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, f1_score, recall_score, precision_score

logger = logging.getLogger()

def fit_logistic_regression():
    """
    Fit a logistic regression on the data to model the review

    Args:
        None
    
    Returns:
        None
    """

    logger.debug("Running the fit_logistic_regression function now")

    #Loading the configuration
    with open(os.path.join("config","config.yml"), "r") as f:
        config = yaml.safe_load(f)

    #Loading and pre processing the data
    logger.debug("Loading and pre processing the data")
    train_df = load_data(config["load_data"]["train_file"])
    train_df = pre_process_data(train_df, resample = True, resample_count = 500000)

    #Defining Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='word', token_pattern=r'[A-Za-z0-9@-]+')),
        ('model', LogisticRegression(random_state=12345, verbose = 1, solver = 'saga')),
    ])

    #Defining parameters to vary
    parameters = {
        'tfidf__max_df': (0.25, 0.5, 0.75),
        'tfidf__max_features': (None, 5000, 10000, 50000),
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'model__C': (0.01, 1, 100)
    }

    scoring_list = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    
    #Performing 5fold CV to determine best hyperparameters
    model = GridSearchCV(pipeline, parameters, cv=5,
                                n_jobs=-1, verbose=1, scoring=scoring_list, refit='f1',)

    t0 = datetime.datetime.now()

    model.fit(train_df["Review"].tolist(), train_df["Ratings"].to_numpy())
    
    logger.info("Grid Search performed in {}".format(str(datetime.datetime.now()-t0)))

    #Saving results
    res_df = pd.DataFrame(model.cv_results_)
    res_df.to_csv(os.path.join(config["summary_stats"]["save_location"], "LogisticRegressionResults.csv"))
    
    #Saving the model
    pickle.dump(model, open(os.path.join(config["models"]["save_location"], "LogisticRegression.pkl"),'wb'))

    return