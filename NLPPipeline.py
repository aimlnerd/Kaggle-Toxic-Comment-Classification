import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from TextPreprocess import TextPreprocess
from DataSplit import DataSplit
from WordEmbedProcess import WordEmbedProcess
from DocClassifier import DocClassifier

'''
                ycol = config['Ycol'],
                xcol = config['Xcol'],
                preprocs=config['analyticSettings']['preprocs'],
                max_features=config['analyticSettings']['max_features'],
                max_len=config['analyticSettings']['max_len'],
                embed_size=config['analyticSettings']['max_len'],
                train_batch_size=config['fitSettings']['train_batch_size'],
                test_batch_size=config['fitSettings']['test_batch_size'],
                epochs=config['fitSettings']['epochs'],
                callbacks=config['fitSettings']['callbacks'],
                best_model_path=config['fitSettings']['best_model_path']
'''

class NLPPipline(BaseEstimator):
    def __init__(
                self,
                ycol,
                xcol,
                valid_size,
                preWordEmbedPath,
                max_features,
                max_len,
                embed_size,
                train_batch_size,
                test_batch_size,
                epochs,
                callbacks=None,
                splittype=["train_test"],
                preprocs=["raw", "fillna", "comment_to_lower", "text_to_seq"],
                best_model_path="best_model.hdf5"
                ):
        super(NLPPipline, self).__init__()
        self.ycol = ycol
        self.xcol = xcol
        self.max_len = max_len
        self.valid_size = valid_size
        self.splittype = splittype
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.best_model_path = best_model_path
        self.textPreprocess = TextPreprocess(preprocs=preprocs, max_features = max_features, max_len = max_len)
        self.dataSplit = DataSplit(valid_size=valid_size, splittype=splittype)
        self.preWordEmbedPath = preWordEmbedPath
        self.wordEmbedProcess = WordEmbedProcess(max_features = max_features, embed_size = embed_size)


    def get_pickable(self):
        return {
                'textPreprocess': self.textPreprocess.get_pickable(),
                'dataSplit': self.dataSplit.get_pickable(),
                'wordEmbedProcess': self.wordEmbedProcess.get_pickable(),
                'docClassifier': self.docClassifier.get_pickable(),
                'ycol': self.ycol,
                'xcol': self.xcol

               }

    def load_pickable(self, pkl):
        self.textPreprocess.load_pickable(pkl['textPreprocess'])
        self.dataSplit.load_pickable(pkl['dataSplit'])
        self.wordEmbedProcess.load_pickable(pkl['wordEmbedProcess'])
        self.docClassifier.load_pickable(pkl['docClassifier'])
        self.ycol = pkl['ycol']
        self.xcol = pkl['xcol']

    def fit(self, train_df):
        """
        send original train data before splitting
        """
        y_train = train_df[self.ycol]
        self.textPreprocess.fit(train_df)
        x_train_pre = self.textPreprocess.transform(train_df)
        x_train, x_valid, y_train, y_valid = self.dataSplit.transform(x_train_pre, y_train)
        glove_wordembed = WordEmbedProcess.load_pretrainedwordembed(self.preWordEmbedPath)  # Note staticmethod is called with classname (ie before initializing class)
        embedding_matrix = self.wordEmbedProcess.fit_transform(pretrainedwordembed=glove_wordembed,
                                                               word_index=self.textPreprocess.word_index)
        self.docClassifier = DocClassifier(embedding_matrix=embedding_matrix,
                                           max_len=self.max_len,
                                           train_batch_size=self.train_batch_size,
                                           test_batch_size=self.test_batch_size,
                                           epochs=self.epochs,
                                           callbacks=self.callbacks,
                                           best_model_path=self.best_model_path)
        self.docClassifier.fit(x_train, x_valid, y_train, y_valid)

    def predict(self, test_df):
        x_test_pre = self.textPreprocess.transform(test_df)
        y_test_pred = self.docClassifier.predict(x_test_pre)
        return y_test_pred


"""
train_df = pd.read_csv('./input/train.csv')
test_df = pd.read_csv('./input/test.csv')

#x = train_df[["comment_text"]]
#y = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

nLPPipline = NLPPipline(
                        ycol=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
                        xcol=["comment_text"],
                        valid_size=0.1,
                        preWordEmbedPath="./input/glove6b50d/glove.6B.50d.txt",
                        splittype=["train_test"],
                        preprocs=["raw", "fillna", "comment_to_lower", "text_to_seq"],
                        max_features=100000,
                        max_len=150,
                        embed_size=50,
                        train_batch_size=256,
                        test_batch_size=1024,
                        epochs=1,
                        callbacks=True,
                        best_model_path="best_model.hdf5"
                        )

nLPPipline.fit(train_df)
y_test_pred = nLPPipline.predict(test_df)
nLPPipline.get_pickable()
"""
        