""" @thnhan
"""
import numpy as np
# from keras.layers import Embedding
from sklearn.preprocessing import RobustScaler

from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, add, Concatenate, Average
from tensorflow.keras.initializers import GlorotUniform

import tensorflow as tf


# def get_layer_embedding(gensim_model, fixlen, name, train_embeddings=False):
#     wv = gensim_model.wv
#     trained_W = wv.vectors
#     trained_W = np.vstack([trained_W, np.zeros(shape=(1, trained_W.shape[1]))])  # them vecto 0 bieu dien pad '_'
#     layer = Embedding(
#         input_dim=trained_W.shape[0], output_dim=trained_W.shape[1],
#         # embeddings_initializer=inits,
#         weights=[trained_W],
#         input_length=fixlen,
#         trainable=train_embeddings,
#         name=name
#     )
#     return layer


# def fe_embedding(emdedding, n_dim, W_regular, drop, n_units, kernel_init, name):
#     dnn = Sequential()
#     # print(n_dim)
#     dnn.add(get_layer_embedding(emdedding,
#                                 n_dim, name=name, train_embeddings=True))
#     dnn.add(Flatten())
#     # dnn.add(BatchNormalization())
#     dnn.add(Dropout(drop))
#
#     dnn.add(Dense(n_units,
#                   kernel_initializer=kernel_init,
#                   activation='relu',
#                   kernel_regularizer=W_regular))
#     dnn.add(BatchNormalization())
#     dnn.add(Dropout(drop))
#
#     dnn.add(Dense(n_units // 2,
#                   kernel_initializer=kernel_init,
#                   activation='relu',
#                   kernel_regularizer=W_regular))
#     dnn.add(BatchNormalization())
#     dnn.add(Dropout(drop))
#
#     dnn.add(Dense(n_units // 4,
#                   kernel_initializer=kernel_init,
#                   activation='relu',
#                   kernel_regularizer=W_regular))
#     dnn.add(BatchNormalization())
#     dnn.add(Dropout(drop))
#
#     dnn.add(Dense(n_units // 8,
#                   kernel_initializer=kernel_init,
#                   activation='relu',
#                   kernel_regularizer=W_regular))
#     dnn.add(BatchNormalization())
#     dnn.add(Dropout(drop))
#
#     # --- Tang 5
#     # dnn.add(Dense(n_units // 16,
#     #               kernel_initializer=kernel_init,
#     #               activation='relu',
#     #               kernel_regularizer=W_regular))
#     # dnn.add(BatchNormalization())
#     # dnn.add(Dropout(drop))
#
#     return dnn


def fe_handcrafted(n_dim, W_regular, drop, n_units, kernel_init):
    dnn = Sequential()

    dnn.add(Dense(n_units, input_dim=n_dim,
                  kernel_initializer=kernel_init,
                  activation='relu',
                  kernel_regularizer=W_regular))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units // 2,
                  kernel_initializer=kernel_init,
                  activation='relu',
                  kernel_regularizer=W_regular))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units // 4,
                  kernel_initializer=kernel_init,
                  activation='relu',
                  kernel_regularizer=W_regular))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units // 8,
                  kernel_initializer=kernel_init,
                  activation='relu',
                  kernel_regularizer=W_regular))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    # dnn.add(Dense(n_units // 16,
    #               kernel_initializer=kernel_init,
    #               activation='relu',
    #               kernel_regularizer=W_regular))
    # dnn.add(BatchNormalization())
    # dnn.add(Dropout(drop))

    return dnn


def hand_net(hand_dim, W_regular, drop=0.5, n_units=1024, seed=123456):
    # ====== To reproduce
    tf.random.set_seed(seed)
    glouni = GlorotUniform(seed=seed)
    # w1 = fe_embedding(gensim_model, dim1,
    #                   W_regular=W_regular,
    #                   drop=drop,
    #                   n_units=n_units,
    #                   kernel_init=glouni, name='emb1')

    s1 = fe_handcrafted(hand_dim,
                        W_regular=W_regular,
                        drop=drop,
                        n_units=n_units,
                        kernel_init=glouni)

    # w2 = fe_embedding(gensim_model, dim1,
    #                   W_regular=W_regular,
    #                   drop=drop,
    #                   n_units=n_units,
    #                   kernel_init=glouni, name='emb2')

    s2 = fe_handcrafted(hand_dim,
                        W_regular=W_regular,
                        drop=drop,
                        n_units=n_units,
                        kernel_init=glouni)
    # print(dim1, dim2)
    in2 = Input(hand_dim)
    in4 = Input(hand_dim)
    x2 = s1(in2)
    x4 = s2(in4)

    # --- Classification
    mer = Average()([x2, x4])
    y = Dense(16, kernel_initializer=glouni,
              activation='relu',  # relu
              kernel_regularizer=W_regular)(mer)
    y = BatchNormalization()(y)
    # y = Dropout(0.25)(y)

    out = Dense(2, kernel_initializer=glouni, activation='softmax')(y)

    final = Model(inputs=[in2, in4], outputs=out)
    return final
