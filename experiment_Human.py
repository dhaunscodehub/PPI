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
from models.df_ppi_model_new_fusion_layer import df_ppi_with_fusion_layer
from protein_descriptors.extracted_handcrafted_features.Human.apaacplus.prepare_handcrafted_HUMAN import \
    prepare_HUMAN_token_and_handfeat_new
from utils.plot_utils.plot_roc_pr_curve import plot_folds
from utils.report_result import print_metrics, my_cv_report

import tensorflow as tf


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


def eval_model(pairs_, labels_, name_APAACplus_):
    start_time = time.time()

    name = name_APAACplus_

    temp = 0
    name_new = name_APAACplus_
    while os.path.exists(name_new):
        temp += 1
        name_new = name + str(temp)
    name = name_new
    os.mkdir(name)

    skf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
    scores = []
    hists = []
    cv_prob_Y, cv_test_y = [], []

    # method_result = dict()
    for ii, (tr_inds, te_inds) in enumerate(skf.split(pairs_, labels_)):
        # if ii in [0, 1, 2, 3]:
        #     continue
        print("\nFold", ii)

        protlen = get_avelen(tr_inds, yeast_dset)
        print("Average length:", protlen)

        tokens_A, hand_A, tokens_B, hand_B, handdim = prepare_HUMAN_token_and_handfeat_new(trained_W1['vocabulary'],
                                                                                           trained_W1[
                                                                                               'embedding_matrix'],
                                                                                           protlen,
                                                                                           yeast_dset,
                                                                                           name_APAACplus_)

        print(handdim)

        tr_A_w2v, tr_A_seq = tokens_A[tr_inds], hand_A[tr_inds]
        tr_B_w2v, tr_B_seq = tokens_B[tr_inds], hand_B[tr_inds]

        te_A_w2v, te_A_seq = tokens_A[te_inds], hand_A[te_inds]
        te_B_w2v, te_B_seq = tokens_B[te_inds], hand_B[te_inds]

        Y = to_categorical(labels_)
        tr_Y, te_Y = Y[tr_inds], Y[te_inds]

        tf.random.set_seed(123456)

        # DEF MODEL
        if os.path.exists(name + str(ii) + '.h5'):
            model = load_model(name + str(ii) + '.h5')
        else:
            model = df_ppi_with_fusion_layer(protlen, handdim, W1=trained_W1['embedding_matrix'], n_units=1024)

        opt = Adam(decay=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Train model
        hist = model.fit([tr_A_w2v, tr_A_seq, tr_B_w2v, tr_B_seq], tr_Y,
                         batch_size=64,  # 64
                         epochs=50,  # 45
                         verbose=0)
        hists.append(hist)

        prob_Y = model.predict([te_A_w2v, te_A_seq, te_B_w2v, te_B_seq])

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

        # -------------------------------------------------------------------------
        # REPORT
        # -------------------------------------------------------------------------
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
    plt.savefig('AUC_AUPR_YeastCore.pdf')
    plt.show()

    return hists


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Global parameters
    # -----------------------------------------------------------------
    print("\nGlobal parameters ...")
    lg = 30;
    print('lg,', lg)
    embedding_size = 32;
    print('embedding_size,', embedding_size)
    path_to_emdeddings = "protein_embedding/doc2vec/d2v_1gram_" + str(embedding_size) + "dim.pkl"
    print('path_to_emdeddings,', path_to_emdeddings)

    for lg in range(30, 31, 2):
        trained_W1 = pickle.load(open(path_to_emdeddings, "rb"))
        trained_W1['vocabulary']['<unk>'] = len(trained_W1['vocabulary'])

        print("Embedding", trained_W1['embedding_matrix'].shape)
        print("Vocabulary", len(trained_W1['vocabulary']))

        yeast_dset, summary = load_raw_dset("data/Human")
        id_pairs = yeast_dset['id_pairs']
        labels = yeast_dset['labels']
        print("Summary:", summary)
        print("Number of pairs:", len(id_pairs))

        name_APAACplus = 'uni_APAACPlus_Human_lg' + str(lg)
        eval_model(id_pairs, labels, name_APAACplus)
