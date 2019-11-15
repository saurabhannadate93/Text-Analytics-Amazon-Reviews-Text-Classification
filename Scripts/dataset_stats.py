"""
Author: Saurabh Annadate

Script to load and perform descriptive analytics on the dataset.

Link to the dataset: https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M (amazon_review_full_csv.tar.gz)

"""

import logging
import os
import pandas as pd
import yaml
import logging
import datetime

from Scripts.text_analytics_helpers import corpus_tokenize
from Scripts.helpers import load_data, update_ratings

logger = logging.getLogger()

def run_count_stats(df, loc, prefix):
    """
    This function runs the data statistics.
    
    Args:
        df (Pandas Dataframe): Input dtaframe for analysis
        loc (String): Save Location for the results
        prefix (String): Prefix for the files
    
    Returns:
        None
    """
    #Record Count:
    nrows = df.shape[0]
    ncols = df.shape[1]

    f = open(os.path.join(loc, prefix + "_summary.txt"),"w+")
    f.write('Number of rows: {}\n\n'.format(nrows))
    f.write('Number of columns: {}\n\n'.format(ncols))
    f.write('Ratings summary:\n')
    f.write(str(df.groupby("Ratings").size().reset_index(name = "Rating_count")))
    f.write("\n\n")
    f.close()

def run_dataset_stats():
    """
    This function loads the data and runs the data statistics.
    
    Args:
        None

    Returns:
        None
    """

    logger.debug("Running the run_dataset_stats function now")

    #Loading the configuration
    with open(os.path.join("config","config.yml"), "r") as f:
        config = yaml.safe_load(f)

    #Loading the training and test dataset
    logger.debug("Loading the training dataset.")
    st_time = datetime.datetime.now()
    train_df = load_data(config["load_data"]["train_file"])
    logger.info("Training dataset loaded in %s", str(datetime.datetime.now() - st_time))

    #Running the summary statistics function    
    logger.debug("Running summary statistics on the raw data")
    run_count_stats(train_df, config["summary_stats"]["save_location"], "train")

    #Selecting only 1,2,4,5 classes and updating classes
    logger.debug("Changing classes from 1,2 to 0 and 4,5 to 1. Removing class 3")
    train_df = update_ratings(train_df)

    #Tokenizing data
    logger.debug("Tokenizing data.")
    train_df = corpus_tokenize(train_df, "Review", "Review_Tokens")

    #Checking the summary of counts of words
    logger.debug("Checking the summary of counts of words.")
    train_df["Token_count"] = train_df["Review_Tokens"].apply(len)

    #Writing the word count summary
    f = open(os.path.join(config["summary_stats"]["save_location"], "token_summary.txt"),"w+")
    f.write('Training words data Summary:\n')
    f.write(str(train_df.groupby("Ratings")["Token_count"].agg(['min', 'max', 'mean'])))
    f.write("\n\n")
    f.close()

    logger.info("Dataset Summary Completed.")

