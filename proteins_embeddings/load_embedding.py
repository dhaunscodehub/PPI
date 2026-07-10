import csv
import pickle

import numpy as np


def load_Doc2vec_embedding(path_to_file):
    data = pickle.load(open(path_to_file, "rb"))
    data['vocabulary']['<unk>'] = len(data['vocabulary'])
    data['W1'] = np.concatenate([data['W1'],
                                 np.zeros((1, data['W1'].shape[1]))], axis=0)

    print("Embedding", data['W1'].shape)
    print("Vocabulary", len(data['vocabulary']))

    return data['W1'], data['vocabulary']


def load_Protvec_embedding(path_to_file="trained_embeddings/by_ProtVec/protVec_100d_3grams.csv"):
    # --------------------------------------------------------------
    # Su dung embeddings da duoc hoc cua Ehsan Asgari (protvec)
    # Source: http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0141287
    # Embedding: https://github.com/ehsanasgari/Deep-Proteomics
    # --------------------------------------------------------------
    ehsanEmbed = []
    with open(path_to_file) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for line in tsvreader:
            ehsanEmbed.append(line[0].split('\t'))
    threemers = [vec[0] for vec in ehsanEmbed]
    embeddingMat = [[float(n) for n in vec[1:]] for vec in ehsanEmbed]
    threemersidx = {}  # generate word to index translation dictionary. Use for kmersdict function arguments.

    for i, kmer in enumerate(threemers):
        threemersidx[kmer] = i

    embeddingMat = np.array(embeddingMat)
    print("\nEmbedding matrix", embeddingMat.shape)

    # trained_W1 = {'W1': , 'vocabulary': threemersidx}

    # print(threemersidx)

    # exit(0)
    # data = pickle.load(open(path_to_file, "rb"))
    # data['vocabulary']['<unk>'] = len(data['vocabulary'])
    # print("Embedding", data['W1'].shape)
    # print("Vocabulary", len(data['vocabulary']))

    return embeddingMat, threemersidx


def load_Bio2vec_embedding(path_to_file="trained_embeddings/by_Bio2vec/Bio2vec_embedding.pkl"):
    data = pickle.load(open(path_to_file, "rb"))
    data['vocabulary']['<unk>'] = len(data['vocabulary'])
    print("Embedding", data['embedding_matrix'].shape)
    print("Vocabulary", len(data['vocabulary']))
    return data['embedding_matrix'], data['vocabulary']
