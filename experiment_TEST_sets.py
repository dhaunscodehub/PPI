import os
import pickle
import time

import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from data.dset_tool import load_raw_dset
from models.df_ppi_model_new_fusion_layer import df_ppi_with_fusion_layer
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
        pos_A_ = np.concatenate([f1_A, f3_A, f2_A], axis=1)
        pos_B_ = np.concatenate([f1_B, f3_B, f2_B], axis=1)

        return pos_A_, pos_B_

    def load_negative():
        NEG_ = pd.read_csv('data/Yeastcore/pairs_neg.csv')
        print(NEG_)

        f1_A, f1_B = ghep_cap(Fvector, NEG_)
        f2_A, f2_B = ghep_cap(LD, NEG_)
        f3_A, f3_B = ghep_cap(APAACplus, NEG_)
        neg_A_ = np.concatenate([f1_A, f3_A, f2_A], axis=1)
        neg_B_ = np.concatenate([f1_B, f3_B, f2_B], axis=1)

        return neg_A_, neg_B_

    Fvector = read_mat_to_feat(path_to_dir + '/uni_Fvector_new.mat')
    print(Fvector)

    LD = read_mat_to_feat(path_to_dir + '/uni_LD.mat')
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


def train_all_YEAST(featute_set: tuple, true_label, name_model, name_APAACplus_):
    start_time = time.time()
    tokens_A, hand_A, tokens_B, hand_B, handdim = prepare_YEAST_token_and_handfeat_new(trained_W1['vocabulary'],
                                                                                       trained_W1[
                                                                                           'embedding_matrix'],
                                                                                       protlen,
                                                                                       yeast_dset, name_APAACplus_)
    print(hand_A.shape)

    # --------------------------------------------------------------
    # Shuffle data
    # --------------------------------------------------------------
    np.random.seed(123456)
    indx = np.arange(len(true_label))
    np.random.shuffle(indx)
    tokens_A, hand_A, tokens_B, hand_B = tokens_A[indx], hand_A[indx], tokens_B[indx], hand_B[indx]
    true_label = true_label[indx]

    true_label = to_categorical(true_label)
    opt = Adam(learning_rate=0.001, decay=0.001)

    model_out = df_ppi_with_fusion_layer(protlen, handdim, W1=trained_W1['embedding_matrix'], n_units=1024)

    model_out.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # Train model
    model_out.fit([tokens_A, hand_A, tokens_B, hand_B], true_label,
                  batch_size=100,
                  epochs=64,
                  shuffle=False,
                  verbose=0)

    print("Training time: ", time.time() - start_time)

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------
    try:
        model_out.save(name_model)
    except:
        new_dir = input("SAVE ERROR, ENTER new folder to save:")
        os.mkdir(new_dir)
        model_out.save(name_model)

    return model_out


def load_Xy_test(path_to_dir, testset_name):
    Fvector_A, Fvector_B = loadmat_TEST_feat(path_to_dir + '/feat_' + testset_name + '_Fvector_new.mat')
    LD_A, LD_B = loadmat_TEST_feat(path_to_dir + '/feat_' + testset_name + '_LD.mat')
    APAACplus_A, APAACplus_B = loadmat_TEST_feat(path_to_dir + '/feat_' + testset_name + '_APAACPlus_lg30.mat')

    feat_A = np.concatenate([Fvector_A, LD_A, APAACplus_A], axis=1)
    feat_B = np.concatenate([Fvector_B, LD_B, APAACplus_B], axis=1)

    print(feat_B.shape)
    true_label = np.array([1] * len(feat_B))

    return feat_A, feat_B, true_label


def run_test(testset_name):
    path_to_dir = "protein_descriptors/extracted_handcrafted_features/extracted_TEST"
    X_test_A, X_test_B, y_test = load_Xy_test(path_to_dir, testset_name)

    test_seq_A, test_seq_B = loadmat_TEST_seq('data/Testsets/' + testset_name + '.mat')

    tokens_A = prot_to_token(trained_W1['vocabulary'], test_seq_A, protlen)
    tokens_B = prot_to_token(trained_W1['vocabulary'], test_seq_B, protlen)

    y_prob = trained_model.predict([tokens_A, X_test_A, tokens_B, X_test_B])

    # ----------------------------------------------------------
    # SAVE
    # ----------------------------------------------------------
    pickle.dump(y_prob, open("results/independent_TEST_full_yeastcore_run9/y_prob_on_" + testset_name + ".pkl", "wb"))

    y_pred = np.argmax(y_prob, axis=1)
    ACC = round(np.sum(y_pred == y_test) / len(y_test), 4)
    print(testset_name, '> Accuracy {}'.format(ACC), np.sum(y_pred == y_test), len(y_test))

    # return A, B, handfeat_dim


if __name__ == "__main__":
    trained_W1 = pickle.load(open("protein_embedding/doc2vec/d2v_1gram_32dim.pkl", "rb"))
    trained_W1['vocabulary']['<unk>'] = len(trained_W1['vocabulary'])
    trained_W1['embedding_matrix'] = np.concatenate([trained_W1['embedding_matrix'],
                                                     np.zeros((1, trained_W1['embedding_matrix'].shape[1]))], axis=0)

    print("Embedding", trained_W1['embedding_matrix'].shape)
    print("Vocabulary", len(trained_W1['vocabulary']))

    yeast_dset, summary = load_raw_dset("data/Yeastcore")
    id_pairs = yeast_dset['id_pairs']
    labels = yeast_dset['labels']
    print("Summary:", summary)
    print("Number of pairs:", len(id_pairs))

    protlen = get_avelen(np.arange(len(id_pairs)), yeast_dset)
    print("Fixed length protein", protlen)

    savedir = "trained_Yeastcore_d2v_1gram_32dim_run9.h5"

    if not os.path.exists(savedir):
        lg = 30
        name_APAACplus = 'uni_APAACPlus_lg' + str(lg)
        trained_model = train_all_YEAST(id_pairs, labels, savedir, name_APAACplus)
    else:
        trained_model = load_model(savedir)

    # ---------------------------------------------------
    # Test on Cross-species
    # ---------------------------------------------------
    run_test("Celeg")
    run_test("Ecoli")
    run_test("Hpylo")
    run_test("Hsapi")
    run_test("Mmusc")

    # ---------------------------------------------------
    # Test on PPI network
    # ---------------------------------------------------
    run_test("Onecore")
    run_test("Wnt")
    run_test("Cancer")
