#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 10:05:14 2018

@author: deepak
"""
import numpy as np
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D
from keras.models import Model, load_model

from RocAucCallback import RocAucCallback
from NLPModel import GRU_CNN_Model


class DocClassifier():
    def __init__(
                    self,
                    embedding_matrix,
                    max_len,
                    train_batch_size,
                    test_batch_size,
                    epochs,
                    callbacks=None,
                    best_model_path="best_model.hdf5"
                ):
        self.embedding_matrix = embedding_matrix
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.best_model_path = best_model_path
        self.max_len = max_len

    def get_pickable(self):
        return {
                'max_len': self.max_len,
                'train_batch_size': self.train_batch_size,
                'test_batch_size': self.test_batch_size,
                'epochs': self.epochs,
                'callbacks': self.callbacks,
                'best_model_path': self.best_model_path
               }

    def load_pickable(self, pkl):
        self.max_len = pkl['max_len']
        self.train_batch_size = pkl['train_batch_size']
        self.test_batch_size = pkl['test_batch_size']
        self.epochs = pkl['epochs']
        self.callbacks = pkl['callbacks']
        self.best_model_path = pkl['best_model_path']

    def callbacks_set(self, x_valid, y_valid):       
        if self.callbacks != None:
            check_point = ModelCheckpoint(self.best_model_path, monitor = "val_loss", verbose = 1,
                                          save_best_only = True, mode = "min")
            ra_val = RocAucCallback(validation_data=(x_valid, y_valid), interval = 1)
            early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)
            self.callbacks = [ra_val, check_point, early_stop]
        return self.callbacks
    
    def fit(self, x_train, x_valid, y_train, y_valid):
        """
        Fits model to data
        """
        if np.any(np.isnan(x_train)):
            print("x_train contains NaNs")
        if np.any(np.isnan(y_train)):
            print("y_train contains NaNs")
        model = GRU_CNN_Model(
                              self.embedding_matrix,
                              max_features=self.embedding_matrix.shape[0],
                              embed_size=self.embedding_matrix.shape[1],
                              input_shape=self.max_len
                              )
        self.model_history = model.fit(x_train, y_train, batch_size = self.train_batch_size,
                                            epochs = self.epochs, validation_data = (x_valid, y_valid), 
                                            verbose = 1, callbacks = self.callbacks_set(x_valid, y_valid)
                                            )
        self.model_fit = load_model(self.best_model_path)
    
    def predict(self,x_test):

        y_test_pred = self.model_fit.predict(x_test, batch_size = self.test_batch_size, verbose = 1)
        return y_test_pred



        
        
        

        