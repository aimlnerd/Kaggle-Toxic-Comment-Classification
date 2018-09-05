import numpy as np
import pandas as pd
import logging
import itertools

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator


class TextPreprocess(BaseEstimator):
    def __init__(self
                 ,preprocs = ['raw']
                 ,max_features = 100000
                 ,max_len = 150
                 ):
        self.preprocs = preprocs # @preprocs.setter gets invoked here
        self.max_features = max_features
        self.max_len = max_len
        self.preprocs_lib_fit = { # fit version of the approaches
            'raw':self.raw_fit
            ,'fillna':self._fillna_fit
            ,'comment_to_lower':self._comment_to_lower_fit
            ,'text_to_seq':self._text_to_seq_fit
        }
        self.preprocs_lib_transform = { # fit version of the approaches
            'raw': self.raw_transform
            ,'fillna': self._fillna_transform
            ,'comment_to_lower': self._comment_to_lower_transform
            ,'text_to_seq': self._text_to_seq_transform
        }
        self.preprocs_info = [] # store info about every filtration for later retrieval and investigation
        super(TextPreprocess, self).__init__()

    def get_pickable(self):
        return {
                'preprocs': self.preprocs,
                'max_features': self.max_features,
                'max_len': self.max_len
               }

    def load_pickable(self, pkl):
        self.preprocs = pkl['preprocs']
        self.max_features = pkl['max_features']
        self.max_len = pkl['max_len']

    @property
    def preprocs(self):
        return self.__preprocs

    @preprocs.setter
    def preprocs(self,value):
        if type(value) is not list:
            logging.error('Preprocessing methods should be passed as a list')
        for v in value:
            if v not in ['raw','fillna','comment_to_lower','text_to_seq']:
                logging.error(f'Preprocessing method {v} is not supported')
            self.__preprocs = value

    def raw_fit(self,df):
        pass

    def raw_transform(self, df):
        return df, {}

    def _fillna_fit(self,df):
        pass # do nothing

    def _fillna_transform(self,df):
        df["comment_text"] = df["comment_text"].fillna("no comment")
        return df, {'#rows with nan imputed': f'{df["comment_text"].isnull().sum()} out of {len(df)}'}

    def _comment_to_lower_fit(self,df):
        pass

    def _comment_to_lower_transform(self,df):
        df["comment_text"] = df["comment_text"].str.lower()
        return df, {}

    def _text_to_seq_fit(self,train):
        self.tk = Tokenizer(num_words=self.max_features,
                            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\'\n', # I added \' it was not there by default 
                            lower=True,
                            split=" ",
                            char_level=False)
        self.tk.fit_on_texts(train["comment_text"])
        self.word_index = self.tk.word_index
        # v.important the values required for transform after fitting should be made as class variables (self) and not returned using return
        # _fit function doesnt return anything but _transform return df and dict (info)

    def _text_to_seq_transform(self, df):
        df["comment_seq"] = self.tk.texts_to_sequences(df["comment_text"])
        x_df = pad_sequences(df["comment_seq"], maxlen=self.max_len)
        return x_df, {}

    def fit(self, x, y=None, **fit_params):
         for f in self.preprocs:
            self.preprocs_lib_fit[f](x)

    def transform(self, x):
        for f in self.preprocs: # filters are appplied sequentially as per order given in self.preprocs
            x, info = self.preprocs_lib_transform[f](x)
            self.preprocs_info.append(info)
        return x

    def fit_transform(self, x, y=None, **fit_params):
        for f in self.preprocs:
            self.preprocs_lib_fit[f](x)
            x, info = self.preprocs_lib_transform[f](x)
            self.preprocs_info.append(info)
            return x

    def print_info(self):
        """ print all the info about applied preprocs """
        print("Preprocessing info")
        for info in self.preprocs_info:
            for k,v in info.items():
                print(k, ":", v)

"""
textPreprocess = TextPreprocess(
                 preprocs = ['raw', 'fillna', 'comment_to_lower', 'text_to_seq']
                 ,max_features = 200000
                 ,max_len = 150
)

textPreprocess.fit(train_df)
train_pre = textPreprocess.transform(train_df)
test_pre = textPreprocess.transform(test_df)
textPreprocess.print_info()
textPreprocess.get_pickable()
"""









