"""@thnhan"""
import pickle

import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import RobustScaler


def prot_to_token(vocal, proteins, prot_len):
    def to_token(prot):
        token = np.array([25] * prot_len)  # 25 la index cua pad
        if len(prot) <= prot_len:
            token[:len(prot)] = [vocal[aa] for aa in prot]
        else:
            token[:len(prot)] = [vocal[aa] for aa in prot[:prot_len]]
        return token

    vocal['_'] = len(vocal) - 1  # them ky tu "_" lam pad
    tokens = np.array(list(map(to_token, proteins)))
    return tokens

# if __name__ == "__main__":
#     size = 16  # 8, 20, 24
#     trained_w2v = Word2Vec.load("trained_W1_" + str(size) + ".wv")
#     trained_W = trained_w2v.wv.vectors
#
#     # trained_W = RobustScaler().fit_transform(trained_W)
#
#     pickle.dump({"W1": trained_W,
#                  "vocabulary": trained_w2v.wv.key_to_index}, open("trained_W1_" + str(size) + ".pkl", "wb"))
