""" @author: thnhan
"""
import os
import pickle
import time

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.model_selection import StratifiedKFold

from data.dset_tool import load_raw_dset
from models.only_embedding_new import only_embedding_net
from utils.plot_utils.plot_roc_pr_curve import plot_folds
from utils.report_result import print_metrics, my_cv_report

from protein_embedding.prepare import prot_to_token


def get_avelen(inds, dset):
    pos_A, pos_B, neg_A, neg_B = dset['seq_pairs']
    pos_AB = np.hstack((pos_A, pos_B))
    neg_AB = np.hstack((neg_A, neg_B))
    prots = np.concatenate((pos_AB, neg_AB), axis=0)
    prots = prots.flatten()
    prots = prots[inds]
    prots = np.unique(prots)
    do_dai = [len(seq) for seq in prots]
    avelen = int(sum(do_dai) / len(do_dai))
    return avelen


# def prepare_YeastCore_token_and_handfeat(vocal, protlen, dset):
#     pos_seq_A, pos_seq_B, neg_seq_A, neg_seq_B = dset['seq_pairs']
#     # pos_hand_A, pos_hand_B, neg_hand_A, neg_hand_B = load_handfeat_YeastCore()
#
#     # --- Lấy đặc trưng bằng Word2vec
#     seqs_A = np.concatenate([pos_seq_A, neg_seq_A], axis=0)
#     seqs_B = np.concatenate([pos_seq_B, neg_seq_B], axis=0)
#     tokens_A = prot_to_token(vocal, seqs_A, protlen)
#     tokens_B = prot_to_token(vocal, seqs_B, protlen)
#     # print('tokens_A', tokens_A.shape)
#
#     # # --- Nối đặc trưng đã được trích xuất vào feature_protein_A, B
#     # hand_A = np.vstack([pos_hand_A, neg_hand_A])
#     # hand_B = np.vstack([pos_hand_B, neg_hand_B])
#     # # print('hand_A', hand_A.shape)
#
#     # --- Concat
#     # seqindex_handfeat = [tokens_A, hand_A, tokens_B, hand_B]
#     # print(seqindex_handfeat)
#     # print(seqindex_handfeat.shape)
#     return tokens_A, tokens_B


def prepare_YeastCore_token(vocal, W1, fixlen_, dset):
    # ----------------------------------------------------------------
    # Load embedding features
    # ----------------------------------------------------------------
    pos_seq_A, pos_seq_B, neg_seq_A, neg_seq_B = dset['seq_pairs']
    seqs_A = np.concatenate([pos_seq_A, neg_seq_A], axis=0)
    seqs_B = np.concatenate([pos_seq_B, neg_seq_B], axis=0)

    tokens_A = prot_to_token(vocal, seqs_A, fixlen_)
    tokens_B = prot_to_token(vocal, seqs_B, fixlen_)

    # print('tokens_A', tokens_A.shape)

    return tokens_A, tokens_B


def eval_model(pairs, prefix_name):
    start_time = time.time()

    temp = 0
    name_new = prefix_name
    while os.path.exists(name_new):
        temp += 1
        name_new = prefix_name + str(temp)
    name = name_new
    os.mkdir(name)

    skf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
    scores = []
    hists = []
    cv_prob_Y, cv_test_y = [], []

    method_result = dict()
    for ii, (tr_inds, te_inds) in enumerate(skf.split(pairs, labels)):
        # if ii in [0,1,2]:
        #     continue
        print("\nFold", ii)
        protlen = get_avelen(tr_inds, yeast_dset)
        print("Average length:", protlen)

        tokens_A, tokens_B = prepare_YeastCore_token(vocabulary,
                                                     embedding_matrix,
                                                     protlen,
                                                     yeast_dset)
        print(tokens_A)
        tr_A_w2v = tokens_A[tr_inds]
        tr_B_w2v = tokens_B[tr_inds]

        te_A_w2v = tokens_A[te_inds]
        te_B_w2v = tokens_B[te_inds]

        Y = to_categorical(labels)
        tr_Y, te_Y = Y[tr_inds], Y[te_inds]

        name = name
        # --- DEF MODEL
        if os.path.exists(name + str(ii) + '.h5'):
            model = load_model(name + str(ii) + '.h5')
        else:
            model = only_embedding_net(tokens_A.shape[1], None,
                                       embedding_matrix,
                                       n_units=1024)
        opt = Adam(decay=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # --- FIT MODEL
        hist = model.fit([tr_A_w2v, tr_B_w2v], tr_Y,
                         batch_size=64,  # 64
                         epochs=50,  # 45
                         verbose=0)
        hists.append(hist)

        prob_Y = model.predict([te_A_w2v, te_B_w2v])

        # -------------------------------------------------------------------------
        # SAVE
        # -------------------------------------------------------------------------
        try:
            # model.save(name + str(ii) + ".h5")
            method_result = {"true_y": np.argmax(te_Y, axis=1), "prob_Y": prob_Y}
            pickle.dump(method_result, open(name + '/y_true_pred_fold' + str(ii) + ".pkl", 'wb'))
            # pickle.dump(hist, open(name + '/hist_fold' + str(ii) + ".pkl", 'wb'))
            np.save(name + '/trainning_history' + str(ii) + '.npy', hist.history)
        except:
            new_dir = input("SAVE ERROR, ENTER new folder to save:")
            os.mkdir(new_dir)

            # model.save(name + str(ii) + ".h5")
            method_result = {"true_y": np.argmax(te_Y, axis=1), "prob_Y": prob_Y}
            pickle.dump(method_result, open(name + '/y_true_pred_fold' + str(ii) + ".pkl", 'wb'))
            # pickle.dump(hist, open(name + '/hist_fold' + str(ii) + ".pkl", 'wb'))
            np.save(name + '/trainning_history' + str(ii) + '.npy', hist.history)

        scr = print_metrics(np.argmax(te_Y, axis=1), prob_Y)
        scores.append(scr)

        cv_prob_Y.append(prob_Y[:, 1])
        cv_test_y.append(np.argmax(te_Y, axis=1))

    # ====== FINAL REPORT
    print("\nFinal scores (mean)")
    scores_array = np.array(scores)
    my_cv_report(scores_array)

    print("Running time", time.time() - start_time)

    plot_folds(plt, cv_test_y, cv_prob_Y)
    plt.tight_layout()
    # plt.savefig('AUC_AUPR_Yeastcore.pdf')
    plt.show()

    return hists


if __name__ == "__main__":
    # --------------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------------
    yeast_dset, summary = load_raw_dset("datasets/Human8161")
    pair_id = yeast_dset['id_pairs']
    labels = yeast_dset['labels']
    print("Summary:", summary)
    print("Number of pairs:", len(pair_id))

    # --------------------------------------------------------------
    # GLOBAL HYPER PARAMETERS
    # --------------------------------------------------------------
    # AAsize = 20  # word (amino acid) size
    # handdim = 650  # handcrafted feature dimension

    # # --------------------------------------------------------------
    # # Protvec
    # # --------------------------------------------------------------
    # embedding_matrix, vocabulary = load_Protvec_embedding("protein_embedding/ProtVec/protVec_100d_3grams.csv")

    # --------------------------------------------------------------
    # Doc2vec
    # --------------------------------------------------------------
    # embedding_matrix, vocabulary = load_Doc2vec_embedding("protein_embedding/Doc2vec/d2v_3gram_128dim.pkl")
    # eval_model(pair_id, "only_embedding_d2v_3gram_128dim")

    for sz in [32]:
        trained_W1 = pickle.load(open("protein_embedding/doc2vec/d2v_1gram_" + str(sz) + "dim.pkl", "rb"))
        trained_W1['vocabulary']['<unk>'] = len(trained_W1['vocabulary'])

        embedding_matrix = trained_W1['embedding_matrix']
        vocabulary = trained_W1['vocabulary']

        print("Embedding", trained_W1['embedding_matrix'].shape)
        print("Vocabulary", len(trained_W1['vocabulary']))

        eval_model(pair_id, "only_embedding_Human_d2v_1gram_" + str(sz) + "dim.pkl")
