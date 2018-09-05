import numpy as np
from sklearn.base import BaseEstimator


class WordEmbedProcess(BaseEstimator):
    def __init__(self,
                 embed_size,
                 max_features
                 ):
        self.max_features = max_features
        self.embed_size = embed_size
        super(WordEmbedProcess, self).__init__()

    def get_pickable(self):
        return {
                'embed_size': self.embed_size,
                'max_features': self.max_features
               }

    def load_pickable(self, pkl):
        self.embed_size = pkl['embed_size']
        self.max_features = pkl['max_features']

    @staticmethod
    def load_pretrainedwordembed(embedding_path):
        pretrained_wordembed = {}
        with open(embedding_path, encoding='utf8') as f:
            for line in f:
                values = line.rstrip().rsplit(' ')
                word = values[0] # 1st value is word
                coefs = np.asarray(values[1:], dtype='float32') # remaining values are coefs
                pretrained_wordembed[word] = coefs
        return pretrained_wordembed

    def fit_transform(self,pretrainedwordembed, word_index, **fit_params):
        """
        Takes max_features number of features from training data and 
        get pretrained glove vectors for those max_features only
        ith index of embedding_matrix denotes ei
        0th index of embedding_matrix will always be 0 because word_index has min index value of 1
        """
        num_words = min(self.max_features, len(word_index)+1)
        embedding_matrix = np.zeros((num_words, self.embed_size))
        for word, i in word_index.items():
            if i >= self.max_features:
                continue
            embedding_vector = pretrainedwordembed.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix



"""
glove_wordembed = WordEmbedProcess.load_pretrainedwordembed(config['preWordEmbedPath']) # Note staticmethod is called with classname (ie before initializing class)
wordembedProcess = WordEmbedProcess(
                        embed_size = 50,
                        max_features = 200000
)

embedding_matrix = wordembedProcess.fit_transform(pretrainedwordembed=glove_wordembed, word_index=textPreprocess.word_index)
"""