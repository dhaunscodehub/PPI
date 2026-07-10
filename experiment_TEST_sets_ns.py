import math
import os
import pickle
import time

import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn import metrics
from sklearn.metrics import f1_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from data.dset_tool import load_raw_dset
from models.df_ppi_model_new_fusion_layer import df_ppi_with_fusion_layer
from protein_descriptors.extracted_handcrafted_features.YEASTcore_ns.load_YEAST_ns_feature import prepare_YEAST_ns, \
    load_YEAST_ns, load_handcrafted_feature
from protein_descriptors.extracted_handcrafted_features.read_mat_file import read_mat_to_feat
from protein_embedding.prepare import prot_to_token
from utils.read_mat_file import loadmat_TEST_feat, loadmat_TEST_seq


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


def load_new_feature(path_to_dir, name_APAACplus_):
    def ghep_cap(uni_, pair_):
        return uni_.loc[pair_.proteinA], uni_.loc[pair_.proteinB]

    def load_positive():
        POS_ = pd.read_csv('data/Yeastcore/pairs_pos.csv')
        print(POS_)
        f1_A, f1_B = ghep_cap(Fvector, POS_)
        f2_A, f2_B = ghep_cap(LD, POS_)
        f3_A, f3_B = ghep_cap(APAACplus, POS_)
        pos_A_ = np.concatenate([f1_A, f2_A, f3_A], axis=1)
        pos_B_ = np.concatenate([f1_B, f2_B, f3_B], axis=1)

        return pos_A_, pos_B_

    def load_negative():
        NEG_ = pd.read_csv('data/Yeastcore/pairs_neg.csv')
        print(NEG_)
        f1_A, f1_B = ghep_cap(Fvector, NEG_)
        f2_A, f2_B = ghep_cap(LD, NEG_)
        f3_A, f3_B = ghep_cap(APAACplus, NEG_)
        neg_A_ = np.concatenate([f1_A, f2_A, f3_A], axis=1)
        neg_B_ = np.concatenate([f1_B, f2_B, f3_B], axis=1)

        return neg_A_, neg_B_

    Fvector = read_mat_to_feat(path_to_dir + '/uni_Fvector_new.mat')
    print(Fvector)

    LD = read_mat_to_feat(path_to_dir + '/uni_LD_NR_YEAST.mat')
    print(LD)

    APAACplus = read_mat_to_feat(path_to_dir + '/' + name_APAACplus_ + '.mat')
    print(APAACplus)

    # print("\n", path_to_dir + '/' + name_APAACplus + '.mat')

    pos_feat_A, pos_feat_B = load_positive()
    neg_feat_A, neg_feat_B = load_negative()

    return pos_feat_A, pos_feat_B, neg_feat_A, neg_feat_B, pos_feat_A.shape[1]


def prepare_YEAST_token_and_handfeat_new(vocal, W1, fixlen_, dset, name_APAACplus_):
    # ----------------------------------------------------------------
    # Load hancrafted features
    # ----------------------------------------------------------------
    p = r"protein_descriptors/extracted_handcrafted_features/Yeastcore/apaacplus"
    pos_hand_A, pos_hand_B, neg_hand_A, neg_hand_B, handfeat_dim = load_new_feature(p, name_APAACplus_)

    hand_A = np.vstack([pos_hand_A, neg_hand_A])
    hand_B = np.vstack([pos_hand_B, neg_hand_B])

    # ----------------------------------------------------------------
    # Load embedding features
    # ----------------------------------------------------------------
    pos_seq_A, pos_seq_B, neg_seq_A, neg_seq_B = dset['seq_pairs']
    seqs_A = np.concatenate([pos_seq_A, neg_seq_A], axis=0)
    seqs_B = np.concatenate([pos_seq_B, neg_seq_B], axis=0)

    tokens_A = prot_to_token(vocal, seqs_A, fixlen_)
    tokens_B = prot_to_token(vocal, seqs_B, fixlen_)

    # print('tokens_A', tokens_A.shape)

    return tokens_A, hand_A, tokens_B, hand_B, handfeat_dim


def train_all_YEAST_ns(featute_set: tuple, true_label, name_model, name_APAACplus_):
    start_time = time.time()
    tokens_A, hand_A, tokens_B, hand_B, handdim = prepare_YEAST_ns(trained_W1['vocabulary'],
                                                                   protlen,
                                                                   yeast_dset, name_APAACplus_)
    print(hand_A.shape)

    # --------------------------------------------------------------
    # Shuffle data
    # --------------------------------------------------------------
    np.random.seed(123456)
    indx_ = np.arange(len(true_label))
    np.random.shuffle(indx_)

    n_tr_samples = int(len(true_label) * 0.8)  # train on 0.9 data
    indx = indx_[:n_tr_samples]
    tr_tokens_A, tr_hand_A, tr_tokens_B, tr_hand_B = tokens_A[indx], hand_A[indx], tokens_B[indx], hand_B[indx]
    tr_true_label = true_label[indx]

    tr_true_label = to_categorical(tr_true_label)
    opt = Adam(learning_rate=0.001, decay=0.001)

    model_out = df_ppi_with_fusion_layer(protlen, handdim, W1=trained_W1['embedding_matrix'], n_units=1024)

    model_out.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Train model
    model_out.fit([tr_tokens_A, tr_hand_A, tr_tokens_B, tr_hand_B], tr_true_label,
                  batch_size=64,  # 64*, 100*, 128
                  epochs=50,  # 64, 100*, 128
                  shuffle=True,
                  verbose=2)

    print("Training time: ", time.time() - start_time)

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------
    try:
        model_out.save_weights(name_model)
    except:
        new_dir = input("SAVE ERROR, ENTER new folder to save:")
        os.mkdir(new_dir)
        model_out.save_weights(name_model)

    # ---------------------------------------------------------------
    # Threshold
    # ---------------------------------------------------------------
    indx = indx_[n_tr_samples:]  # holdout set
    h_tokens_A, h_hand_A, h_tokens_B, h_hand_B = tokens_A[indx], hand_A[indx], tokens_B[indx], hand_B[indx]

    hold_targets = true_label[indx]
    hold_targets = to_categorical(hold_targets)

    # print(hold_targets)
    prob_hold = model_out.predict([h_tokens_A, h_hand_A, h_tokens_B, h_hand_B])

    # print(prob_hold)
    #
    # pr, re, threshold_pr = metrics.precision_recall_curve(hold_targets[:, 1], prob_hold[:, 1])
    # aupr = metrics.auc(re, pr)
    # f1_s = [2 * p * r / (p + r) for p, r in zip(pr, re)]
    # opt_f1_index = np.argmax(f1_s)
    # opt_threshold_pr = threshold_pr[opt_f1_index]
    # print(opt_threshold_pr)
    # print(f1_s[opt_f1_index])

    def to_labels(pos_probs, threshold):
        return (pos_probs > threshold).astype('int')

    # define thresholds
    thresholds = np.arange(0.5, 1, 0.001)
    # evaluate each threshold
    scores = [f1_score(hold_targets[:, 1], to_labels(prob_hold[:, 1], t)) for t in thresholds]
    # get best threshold
    ix = np.argmax(scores)
    print('\nThreshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

    return model_out, thresholds[ix]


def load_Xy_test(path_to_dir, testset_name):
    Fvector_A, Fvector_B = loadmat_TEST_feat(path_to_dir + '/feat_' + testset_name + '_Fvector_new.mat')
    LD_A, LD_B = loadmat_TEST_feat(path_to_dir + '/feat_' + testset_name + '_LD.mat')
    APAACplus_A, APAACplus_B = loadmat_TEST_feat(path_to_dir + '/feat_' + testset_name + '_APAACPlus_lg30.mat')

    feat_A = np.concatenate([Fvector_A, LD_A, APAACplus_A], axis=1)
    feat_B = np.concatenate([Fvector_B, LD_B, APAACplus_B], axis=1)

    print(feat_B.shape)
    true_label_ = np.array([1] * len(feat_B))

    return feat_A, feat_B, true_label_


def run_test(testset_name, cls_thrs):
    path_to_dir = "protein_descriptors/extracted_handcrafted_features/extracted_TEST"
    X_test_A, X_test_B, y_test = load_Xy_test(path_to_dir, testset_name)

    test_seq_A, test_seq_B = loadmat_TEST_seq('data/Testsets/' + testset_name + '.mat')

    tokens_A = prot_to_token(trained_W1['vocabulary'], test_seq_A, protlen)
    tokens_B = prot_to_token(trained_W1['vocabulary'], test_seq_B, protlen)

    y_prob = trained_model.predict([tokens_A, X_test_A, tokens_B, X_test_B])
    # print(y_prob)

    # ----------------------------------------------------------
    # SAVE
    # ----------------------------------------------------------
    pickle.dump(y_prob, open("y_prob_on_" + testset_name + ".pkl", "wb"))

    y_pred = y_prob[:, 1] > cls_thrs
    ACC = round(np.sum(y_pred == y_test) / len(y_test), 4)
    print(testset_name, '> Accuracy {}'.format(ACC), np.sum(y_pred == y_test), len(y_test))


if __name__ == "__main__":
    # ---------------------------------------------------------------------------------
    # Use amino acid embeddings pretrain on SwissProt database after removing
    # protein sequences that have similarity >=40% with protein sequences in
    # indepedent test sets
    # ----------------------------------------------------------------------------------
    trained_W1 = pickle.load(open("protein_embedding/doc2vec/d2v_1gram_32dim_SwissProt_ns.pkl", "rb"))
    trained_W1['vocabulary']['<unk>'] = len(trained_W1['vocabulary'])
    trained_W1['embedding_matrix'] = np.concatenate([trained_W1['embedding_matrix'],
                                                     np.zeros((1, trained_W1['embedding_matrix'].shape[1]))], axis=0)

    print("Embedding", trained_W1['embedding_matrix'].shape)
    print("Vocabulary", len(trained_W1['vocabulary']))

    # yeast_dset, summary = load_raw_dset("data/Yeastcore")
    yeast_dset, summary = load_YEAST_ns("data/YEASTcore_ns")
    id_pairs = yeast_dset['id_pairs']
    labels = yeast_dset['labels']
    print("Summary:", summary)
    print("Number of pairs:", len(id_pairs))

    protlen = get_avelen(np.arange(len(id_pairs)), yeast_dset)
    print("Fixed length protein", protlen)

    savedir = "trained_d2v_1gram_32dim_SwissProt_ns.h5"

    if not os.path.exists(savedir):
        lg = 30
        name_APAACplus = 'uni_APAACPlus_lg' + str(lg)
        trained_model, opt_thrs = train_all_YEAST_ns(id_pairs, labels, savedir, name_APAACplus)
        print(opt_thrs)
        with open('results/opt_thrs.txt', 'w+') as fo:
            fo.write(str(opt_thrs) + '\n')
    else:
        trained_model = load_model(savedir)
        dir_p = os.path.dirname(savedir)
        fi = open("results/opt_thrs.txt", 'r')
        opt_thrs = float(fi.read())

    # ---------------------------------------------------
    # Test on PPI network
    # ---------------------------------------------------
    run_test("Onecore", opt_thrs)
    run_test("Wnt", opt_thrs)
    run_test("Cancer", opt_thrs)
