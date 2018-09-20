from keras.optimizers import Adam, RMSprop
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, BatchNormalization, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.models import Model


def GRU_CNN_Model(embedding_matrix,
                  max_features,
                  embed_size,
                  input_shape
                  ):
    inp = Input(shape = (input_shape,))
    x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(rate=0.2)(x)
    # return sequences return the hidden state output for each input time step.
    # return state returns the hidden state output and cell state for the last input time step.
    x = Bidirectional(GRU(units=128, return_sequences=True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])

    x = Dense(6, activation = "sigmoid")(x)
    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = 1e-3, decay = 0), metrics = ["accuracy"])
    return model

