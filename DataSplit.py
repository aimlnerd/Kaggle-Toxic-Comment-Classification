import numpy as np
import pandas as pd
import logging
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split


class DataSplit(BaseEstimator):
    def __init__(self,
                 valid_size,
                 splittype = ["train_test"]
                ):
        super(DataSplit, self).__init__()
        self.valid_size = valid_size
        self.splittype = splittype
        self.splittype_lib_fit = { # fit version of the approaches
            'raw':self._raw_fit
            ,'train_test':self._train_test_fit

        }
        self.splittype_lib_transform = { # fit version of the approaches
            'raw': self._raw_transform
            ,'train_test': self._train_test_transform
        }

    def get_pickable(self):
        return {
                'valid_size': self.valid_size,
                'splittype': self.splittype
               }

    def load_pickable(self, pkl):
        self.valid_size = pkl['valid_size']
        self.splittype = pkl['splittype']

    @property
    def splittype(self):
        return self.__splittype

    @splittype.setter
    def splittype(self,value):
        if type(value) is not list:
            logging.error('Preprocessing methods should be passed as a list')
        for v in value:
            if v not in ['train_test']:
                logging.error(f'Preprocessing method {v} is not supported')
            self.__splittype = value

    def _raw_fit(self, x, y, **fit_params):
        pass

    def _raw_transform(self, x, y, **fit_params):
        return x, y

    def _train_test_fit(self, x, y, **fit_params):
        pass

    def _train_test_transform(self, x, y, **fit_params):
        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=self.valid_size)
        return x_train, x_valid, y_train, y_valid

    def fit(self, x, y, **fit_params):
             for f in self.splittype:
                self.splittype_lib_fit[f](x, y)

    def transform(self, x, y, **fit_params):
        for f in self.splittype: # filters are appplied sequentially as per order given in self.preprocs
            x_train, x_valid, y_train, y_valid = self.splittype_lib_transform[f](x, y)
        return x_train, x_valid, y_train, y_valid

'''
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

x = train_df[["comment_text"]]
y = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]
dataSplit=DataSplit(valid_size=0.1,
                    splittype=["train_test"])
a,b,c,d = dataSplit.transform(x,y)
'''