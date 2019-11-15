"""
Author: Saurabh Annadate

This script contains all helper functions
"""

import logging
import os
import pandas as pd
import logging
import datetime

logger = logging.getLogger()

def load_data(loc):
    """
    Loads the data as provided in the argument

    Args:
        loc (String): Location of the file name

    Returns:
        Pandas Dataframe with the file loaded
    """

    df = pd.read_csv(loc, header=None)
    
    df.rename({
        0: "Ratings",
        1: "Review_head",
        2: "Review"
    }, inplace = True, axis = 1)

    return df



def update_ratings(df):
    """
    Filters for required ratings and updates 1,2 to 0 and 4,5 to 1

    Args:
        df (Pandas Dataframe): Input dataframe

    Returns:
        Pandas Dataframe with the ratings updated
    """

    #Selecting only 1,2,4,5 classes and updating classes
    df = df.loc[df.Ratings != 3]
    df.loc[(df.Ratings==1) | (df.Ratings==2),["Ratings"]] = 0
    df.loc[(df.Ratings==4) | (df.Ratings==5),["Ratings"]] = 1

    return df

