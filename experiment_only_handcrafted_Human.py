""" @author: thnhan
"""
import os
import pickle
import time

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.model_selection import StratifiedKFold

from data.dset_tool import load_raw_dset
from models.only_handcrafted import hand_net
from protein_descriptors.extracted_handcrafted_features.Human.apaacplus.prepare_handcrafted_HUMAN import load_new_feature
from utils.plot_utils.plot_roc_pr_curve import plot_folds
from utils.report_result import print_metrics, my_cv_report
import matplotlib.pyplot as plt


def eval_model(prefix_name):
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
    cv_prob_y, cv_test_y = [], []
    method_result = dict()
    for ii, (tr_inds, te_inds) in enumerate(skf.split(pairs, labels)):
        print("\nFold:", ii)

        tr_hand_A, tr_hand_B = hand_A[tr_inds], hand_B[tr_inds]
        te_hand_A, te_hand_B = hand_A[te_inds], hand_B[te_inds]

        print("Hancrafted features", tr_hand_A.shape)

        Y = to_categorical(labels)
        tr_Y, te_Y = Y[tr_inds], Y[te_inds]

        if os.path.exists(name + str(ii) + '.h5'):
            model = load_model(name + str(ii) + '.h5')
        else:
            model = hand_net(hand_dim=handdim, W_regular=None, n_units=1024)

        opt = Adam(decay=0.001)
        model.compile(loss=loss_fn, optimizer=opt, metrics=['accuracy'])

        hist = model.fit([tr_hand_A, tr_hand_B], tr_Y,
                         batch_size=64,
                         epochs=50,
                         verbose=0)
        hists.append(hist)

        prob_Y = model.predict([te_hand_A, te_hand_B])

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

        cv_prob_y.append(prob_Y[:, 1])
        cv_test_y.append(np.argmax(te_Y, axis=1))

    # ====== FINAL REPORT
    print("\nFinal scores (mean)")
    scores_array = np.array(scores)
    my_cv_report(scores_array)

    print("Running time", time.time() - start_time)

    plot_folds(plt, cv_test_y, cv_prob_y)
    plt.tight_layout()
    # plt.savefig('AUC_AUPR_YeastFull.pdf')
    plt.show()
    print("Fold", time.time() - start_time)
    return hists


def load_handfeat_YeastCore():
    # ----------------------------------------------------------------
    # Load hancrafted features
    # ----------------------------------------------------------------
    p = r"protein_descriptors/extracted_handcrafted_features/Human/apaacplus"
    pos_hand_A, pos_hand_B, neg_hand_A, neg_hand_B, handfeat_dim = load_new_feature(p)

    hand_A = np.vstack([pos_hand_A, neg_hand_A])
    hand_B = np.vstack([pos_hand_B, neg_hand_B])

    return hand_A, hand_B, handfeat_dim


if __name__ == "__main__":
    # --------------------------------------------------------------
    # GLOBAL HYPER PARAMETERS
    # --------------------------------------------------------------
    loss_fn = 'categorical_crossentropy'

    # --------------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------------
    yeast_dset, summary = load_raw_dset("datasets/Human8161")
    pairs = yeast_dset['id_pairs']
    labels = yeast_dset['labels']
    print("Summary:", summary)
    print("Number of pairs:", len(pairs))

    # --------------------------------------------------------------
    # Load feature
    # --------------------------------------------------------------
    hand_A, hand_B, handdim = load_handfeat_YeastCore()

    eval_model('only_handcrafted_Human')
