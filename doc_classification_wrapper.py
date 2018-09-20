
import time
from sklearn.model_selection import train_test_split
import sys, os, re, csv, codecs, numpy as np, pandas as pd
np.random.seed(32)
#os.environ["OMP_NUM_THREADS"] = "4"

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer

import logging
import traceback
import json
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback

start_time = time.time()

def read_data_from_files(config, log):
    try:
        train_df = pd.read_csv(filepath_or_buffer=config['trainPath'], usecols = config['id']+config['Xcol']+config['Ycol'])
        test_df = pd.read_csv(filepath_or_buffer=config['testPath'], usecols = config['id']+config['Xcol'])
    except:
        log.info(f"Error: Failed to import data from {config['trainPath']} and {config['testPath']}")
        error_message = traceback.format_exc()
        log.error(error_message)
    return train_df, test_df

def load_saved_model(config):
    pass

def train():
    pass

def predict():
    pass

def generate_model_metrics():
    pass


if __name__ == '__main__':
    #jsonfile = sys.argv[1]
    jsonfile = "./config/fit.json"
    logFileName = jsonfile.split("/")[-1].split(".")[0] + '.log'
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG,
                        handlers=[
                                logging.FileHandler(filename=logFileName, mode='w'),
                                logging.StreamHandler()
                                ]
                        )
    log_instance = logging.getLogger('root')
    techname = 'LSTM_Doc_Classification'

    with open(jsonfile,'r') as json_file:
        config = json.load(json_file)
    log_instance.info('Config json successfully read')

    train_df, test_df = read_data_from_files(config, log_instance)
    log_instance.info('Data files read in successfully')

