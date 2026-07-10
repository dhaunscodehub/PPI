""" @thnhan """
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, add, Concatenate, Average, Lambda
from tensorflow.keras.initializers import GlorotUniform, RandomUniform

import tensorflow as tf

# ---------------------------------------------------
# GLOBAL setting for model
# To reproduce the same result
# ---------------------------------------------------
tf.random.set_seed(123456)
glouni = GlorotUniform(seed=123456)


def embedding_layer(trained_W_protvec, fixlen, name, train_embeddings=False):
    layer = Embedding(
        input_dim=trained_W_protvec.shape[0], output_dim=trained_W_protvec.shape[1],
        weights=[trained_W_protvec],
        input_length=fixlen,
        trainable=train_embeddings,
        name=name
    )
    return layer


def fe_embedding(embedding, n_dim, drop, n_units, kernel_init, name):
    dnn = Sequential()
    # print(n_dim)
    dnn.add(embedding_layer(embedding,
                            n_dim, name=name, train_embeddings=True))
    dnn.add(Flatten())
    # dnn.add(BatchNormalization())  # for test, da thu, nen tat
    dnn.add(Dropout(drop))

    dnn.add(Dense(2048,
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

    dnn.add(Dense(n_units // 8,
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


def fe_handcrafted(n_dim, drop, n_units, kernel_init):
    dnn = Sequential()

    dnn.add(Dense(n_units, input_dim=n_dim,
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

    dnn.add(Dense(n_units // 8,
                  kernel_initializer=kernel_init,
                  activation='relu'))
    dnn.add(BatchNormalization())
    dnn.add(Dropout(drop))

    # dnn.add(Dense(n_units // 16,
    #               kernel_initializer=kernel_init,
    #               activation='relu',
    #               kernel_regularizer=W_regular))
    # dnn.add(BatchNormalization())
    # dnn.add(Dropout(drop))

    return dnn


def fusion_layer(x, omega=0.5):
    return omega * x[0] + (1 - omega) * x[1]


def df_ppi_with_fusion_layer(dim1, dim2, W1, drop=0.5, n_units=1024, seed=123456):
    w1 = fe_embedding(W1, dim1,
                      drop=drop,
                      n_units=n_units,
                      kernel_init=glouni, name='emb1')

    s1 = fe_handcrafted(dim2, drop=drop,
                        n_units=n_units,
                        kernel_init=glouni)

    w2 = fe_embedding(W1, dim1,
                      drop=drop,
                      n_units=n_units,
                      kernel_init=glouni, name='emb2')

    s2 = fe_handcrafted(dim2, drop=drop,
                        n_units=n_units,
                        kernel_init=glouni)
    # print(dim1, dim2)
    in1, in2 = Input(dim1), Input(dim2)
    in3, in4 = Input(dim1), Input(dim2)
    x1, x2 = w1(in1), s1(in2)
    x3, x4 = w2(in3), s2(in4)

    # Fusion layer 1
    mer1 = Lambda(fusion_layer)([x1, x2])
    den1 = Dense(8, kernel_initializer=glouni,  # 5, 8*, 16
                 activation='relu')(mer1)
    den1 = BatchNormalization()(den1)
    out1 = Dropout(0.5)(den1)

    # Fusion layer 2
    mer2 = Lambda(fusion_layer)([x3, x4])
    den2 = Dense(8, kernel_initializer=glouni,  # 5, 8*, 16
                 activation='relu')(mer2)
    den2 = BatchNormalization()(den2)
    out2 = Dropout(0.5)(den2)

    # Classification
    mer = Average()([out1, out2])  # Average
    y = Dense(16, kernel_initializer=glouni,  # 4, 8, 16*
              activation='relu')(mer)
    y = BatchNormalization()(y)

    out = Dense(2, kernel_initializer=glouni, activation='softmax')(y)

    final = Model(inputs=[in1, in2, in3, in4], outputs=out)
    # print(final.summary())
    # tf.keras.utils.plot_model(final, "model.png", show_shapes=True)
    return final
