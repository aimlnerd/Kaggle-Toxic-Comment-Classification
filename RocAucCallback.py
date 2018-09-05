#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 10:13:42 2018

@author: deepak
"""
import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


class RocAucCallback(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

"""
Sample output with AUC calculated for every epoch


Train on 143613 samples, validate on 15958 samples
Epoch 1/4
143613/143613 [==============================] - 272s 2ms/step - loss: 0.0740 - acc: 0.9749 - val_loss: 0.0549 - val_acc: 0.9800

 ROC-AUC - epoch: 1 - score: 0.971230

Epoch 00001: val_loss improved from inf to 0.05489, saving model to best_model.hdf5
Epoch 2/4
143613/143613 [==============================] - 256s 2ms/step - loss: 0.0566 - acc: 0.9796 - val_loss: 0.0511 - val_acc: 0.9809

 ROC-AUC - epoch: 2 - score: 0.978396

Epoch 00002: val_loss improved from 0.05489 to 0.05112, saving model to best_model.hdf5
Epoch 3/4
143613/143613 [==============================] - 265s 2ms/step - loss: 0.0519 - acc: 0.9810 - val_loss: 0.0487 - val_acc: 0.9819

 ROC-AUC - epoch: 3 - score: 0.981291

Epoch 00003: val_loss improved from 0.05112 to 0.04867, saving model to best_model.hdf5
Epoch 4/4
143613/143613 [==============================] - 267s 2ms/step - loss: 0.0494 - acc: 0.9817 - val_loss: 0.0496 - val_acc: 0.9819

 ROC-AUC - epoch: 4 - score: 0.981320

Epoch 00004: val_loss did not improve
"""