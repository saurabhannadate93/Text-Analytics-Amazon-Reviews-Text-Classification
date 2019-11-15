"""
Author: Saurabh Annadate

An implementation of text classifier using word-level embeddings
and a convolutional neural network.

"""
import os
import yaml
import logging
import math
import pickle

from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout
from keras.layers.pooling import MaxPooling1D
from keras.layers.convolutional import Conv1D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import gensim
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import keras

from Scripts.text_analytics_helpers import pre_process_data
from Scripts.helpers import load_data

from Scripts.text_analytics_helpers import corpus_tokenize, texts_to_indices, token_to_index

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logger = logging.getLogger()

# model hyper parameters
EMBEDDING_DIM = 100
SEQUENCE_LENGTH_PERCENTILE = 99
n_layers = 6
n_fc_layers = 2
hidden_units = 256
batch_size = 200
pretrained_embedding = False
# if we have pre-trained embeddings, specify if they are static or non-static embeddings
TRAINABLE_EMBEDDINGS = True
patience = 3
dropout_rate = 0.3
n_filters = 256
window_size = 6
dense_activation = "relu"
l2_penalty = 0.0003
epochs = 25
VALIDATION_SPLIT = 0.1

def train(train_texts, train_labels, dictionary, model_file=None, EMBEDDINGS_MODEL_FILE=None):
    """
    Train a word-level CNN text classifier.
    :param train_texts: tokenized and normalized texts, a list of token lists, [['sentence', 'blah', 'blah'], ['sentence', '2'], .....]
    :param train_labels: the label for each train text
    :param dictionary: A gensim dictionary object for the training text tokens
    :param model_file: An optional output location for the ML model file
    :param EMBEDDINGS_MODEL_FILE: An optional location for pre-trained word embeddings file location
    :return: the produced keras model, the validation accuracy, and the size of the training examples
    """
    assert len(train_texts)==len(train_labels)
    
    #Loading the configuration
    with open(os.path.join("config","config.yml"), "r") as f:
        config = yaml.safe_load(f)

    # compute the max sequence length
    lengths=list(map(lambda x: len(x), train_texts))
    MAX_SEQUENCE_LENGTH = int(np.percentile(np.array(lengths), SEQUENCE_LENGTH_PERCENTILE))
    
    #Saving Max Sequence Length
    with open(os.path.join(config["models"]["save_location"],'cnn_MSL'), 'wb') as f:
        pickle.dump(MAX_SEQUENCE_LENGTH, f)
    
    # convert all texts to dictionary indices
    train_texts_indices = list(map(lambda x: texts_to_indices(x, dictionary), train_texts))
    
    # pad or truncate the texts
    x_data = pad_sequences(train_texts_indices, maxlen=int(MAX_SEQUENCE_LENGTH))

    # convert the train labels to one-hot encoded vectors
    train_labels = keras.utils.to_categorical(train_labels)
    y_data = train_labels

    model = Sequential()

    # create embeddings matrix from word2vec pre-trained embeddings, if provided
    if pretrained_embedding:
        embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(EMBEDDINGS_MODEL_FILE, binary=True)
        embedding_matrix = np.zeros((len(dictionary) + 1, EMBEDDING_DIM))
        for word, i in dictionary.token2id.items():
            embedding_vector = embeddings_index[word] if word in embeddings_index else None
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        
        model.add(Embedding(len(dictionary) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=TRAINABLE_EMBEDDINGS))
    else:
        model.add(Embedding(len(dictionary) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH))
    
    # add a 1 dimensional conv layer
    for _ in range(math.ceil(n_layers/2)):
        model.add(Dropout(dropout_rate))
        model.add(Conv1D(filters=n_filters,
                        kernel_size=window_size,
                        activation='relu'))
    
    for _ in range(n_layers - math.ceil(n_layers/2)):
        model.add(Dropout(dropout_rate))
        model.add(Conv1D(filters=n_filters,
                        kernel_size=math.ceil(window_size/2),
                        activation='relu'))
    
    # add a max pooling layer
    model.add(MaxPooling1D(2))
    model.add(Flatten())

    # add 0 or more fully connected layers with drop out
    for _ in range(n_fc_layers):
        model.add(Dropout(dropout_rate))
        model.add(Dense(hidden_units,
                        activation=dense_activation,
                        kernel_regularizer=l2(l2_penalty),
                        bias_regularizer=l2(l2_penalty),
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros'))

    # add the last fully connected layer with softmax activation
    model.add(Dropout(dropout_rate))
    model.add(Dense(len(train_labels[0]),
                    activation='softmax',
                    kernel_regularizer=l2(l2_penalty),
                    bias_regularizer=l2(l2_penalty),
                    kernel_initializer='glorot_uniform',
                    bias_initializer='zeros'))

    # compile the model, provide an optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # print a summary
    print(model.summary())
    
    # train the model with early stopping
    early_stopping = EarlyStopping(patience=patience)
    Y = np.array(y_data)

    fit = model.fit(x_data,
                    Y,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=VALIDATION_SPLIT,
                    verbose=1,
                    callbacks=[early_stopping])

    logger.info(fit.history.keys())
    val_accuracy = fit.history['acc'][-1]
    logger.info(val_accuracy)
    
    # save the model
    if model_file:
        model.save(model_file)
    
    return model, val_accuracy, len(train_labels)


def fit_cnn():
    """
    Fits a CNN on the data to model the review

    Args:
        None
    
    Returns:
        None
    """
    
    logger.debug("Running the fit_cnn function now")

    #Loading the configuration
    with open(os.path.join("config","config.yml"), "r") as f:
        config = yaml.safe_load(f)

    #Loading and pre processing the data
    logger.debug("Loading and pre processing the data")
    train_df = load_data(config["load_data"]["train_file"])
    train_df = pre_process_data(train_df, resample = True, resample_count = 500000)

    #Generating texts and labels
    train_df = corpus_tokenize(train_df, "Review", "Review_Tokens")
    texts = train_df['Review_Tokens'].tolist()
    labels = train_df['Ratings'].tolist()

    mydict = gensim.corpora.Dictionary(texts)
    mydict.save(os.path.join(config["summary_stats"]["save_location"], "amazon.dict"))
    
    train(texts, labels, mydict, model_file=os.path.join(config["models"]["save_location"],'amazon_cnn.model'))

    return
