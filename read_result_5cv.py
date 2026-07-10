import os
import pickle
from datetime import time

import numpy as np
from matplotlib import pyplot as plt

from utils.plot_utils.plot_roc_pr_curve import plot_folds
from utils.report_result import print_metrics, my_cv_report

name = ""

print("\n5-fold cross-validation on Yeastcore\n")

scores = []
hists = []
cv_prob_Y, cv_test_y = [], []

# n = 'results_on_YeastCore/y_true_pred_fold'
n = 'uni_APAACPlus_lg30/y_true_pred_fold'
for ii in range(5):
    method_result = pickle.load(open(n + str(ii) + '.pkl', 'rb'))
    true_y = method_result['true_y']
    prob_Y = method_result['prob_Y']

    cv_prob_Y.append(prob_Y[:, 1])
    cv_test_y.append(true_y)

    # -------------------------------------------------------------------------
    # REPORT Fold
    # -------------------------------------------------------------------------
    scr = print_metrics(true_y, prob_Y, verbose=1)
    scores.append(scr)

# -------------------------------------------------------------------------
# REPORT
# -------------------------------------------------------------------------
scores_array = np.array(scores)
my_cv_report(scores_array)

plot_folds(plt, cv_test_y, cv_prob_Y)
plt.tight_layout()
# plt.savefig('AUC_AUPR_YeastCore.pdf')
plt.show()


# %%
print("\n5-fold cross-validation on Human\n")

scores = []
hists = []
cv_prob_Y, cv_test_y = [], []

# n = 'results_on_Human/y_true_pred_fold'
n = 'uni_APAACPlus_Human_lg30/y_true_pred_fold'
for ii in range(5):
    method_result = pickle.load(open(n + str(ii) + '.pkl', 'rb'))
    true_y = method_result['true_y']
    prob_Y = method_result['prob_Y']

    cv_prob_Y.append(prob_Y[:, 1])
    cv_test_y.append(true_y)

    # -------------------------------------------------------------------------
    # REPORT Fold
    # -------------------------------------------------------------------------
    scr = print_metrics(true_y, prob_Y, verbose=0)
    scores.append(scr)

# -------------------------------------------------------------------------
# REPORT
# -------------------------------------------------------------------------
scores_array = np.array(scores)
my_cv_report(scores_array)

plot_folds(plt, cv_test_y, cv_prob_Y)
plt.tight_layout()
# plt.savefig('AUC_AUPR_YeastCore.pdf')
plt.show()
