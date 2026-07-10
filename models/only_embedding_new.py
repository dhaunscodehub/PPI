""" @thnhan
"""
import pickle

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Average, Dense, BatchNormalization, Dropout, Flatten, Embedding, add, Concatenate
from tensorflow.keras.initializers import GlorotUniform, GlorotNormal
from sklearn.preprocessing import StandardScaler, RobustScaler
import tensorflow as tf


# ====== To reproduce the same result
tf.random.set_seed(123456)
glouni = GlorotUniform(seed=123456)


def embedding_layer(trained_embedding, fixlen, name, train_embeddings=True):
    # Su dung ma tran nhung da duoc hoc tu mo hinh Prot2Vec, Doc2vec, ProteinBert, ...
    # Ki tu pad la '<unk>' cung duoc chuyen thanh mot vector 100 chieu.

    layer = Embedding(
        input_dim=trained_embedding.shape[0],
        output_dim=trained_embedding.shape[1],
        weights=[trained_embedding],
        input_length=fixlen,
        trainable=train_embeddings,
        # embeddings_initializer=RandomUniform(minval=-0.05, maxval=0.05, seed=42),
        name=name
    )
    return layer


def fe_embedding(embedding_matrix_in, handcrafted_dim, drop, n_units, kernel_init, name):
    dnn = Sequential()
    # print(n_dim)
    dnn.add(embedding_layer(embedding_matrix_in,
                            handcrafted_dim,
                            name=name, train_embeddings=True))
    dnn.add(Flatten())

    # dnn.add(BatchNormalization())
    # dnn.add(Dropout(drop))  # co the bat len

    n_units = 2048
    dnn.add(Dense(n_units,
                  kernel_initializer=kernel_init,
                  activation='relu'))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units // 2,
                  kernel_initializer=kernel_init,
                  activation='relu'))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(n_units // 4,
                  kernel_initializer=kernel_init,
                  activation='relu'))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    dnn.add(Dense(128, # n_units // 8,
                  kernel_initializer=kernel_init,
                  activation='relu'))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    # --- Tang 5
    # dnn.add(Dense(n_units // 16,
    #               kernel_initializer=kernel_init,
    #               activation='relu',
    #               kernel_regularizer=W_regular))
    # dnn.add(BatchNormalization())
    # dnn.add(Dropout(drop))

    return dnn


def only_embedding_net(fixlen_in, handcrafted_dim, embedding_matrix_in, n_units, drop=0.5):
    w1 = fe_embedding(embedding_matrix_in,
                      fixlen_in,
                      drop=drop,
                      n_units=n_units,
                      kernel_init=glouni, name='emb1')

    w2 = fe_embedding(embedding_matrix_in,
                      fixlen_in,
                      drop=drop,
                      n_units=n_units,
                      kernel_init=glouni, name='emb2')

    in1 = Input(fixlen_in)
    in3 = Input(fixlen_in)
    x1 = w1(in1)
    x3 = w2(in3)

    # --- Classification
    mer = Average()([x1, x3])
    y = Dense(16, kernel_initializer=glouni,  # 4, 8
              activation='relu')(mer)
    y = BatchNormalization()(y)
    # y = Dropout(0.25)(y)

    out = Dense(2, kernel_initializer=glouni, activation='softmax')(y)

    final = Model(inputs=[in1, in3], outputs=out)

    # print(final.summary())
    # tf.keras.utils.plot_model(final, "model.png", show_shapes=True)

    return final
