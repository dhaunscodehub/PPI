"""
@author: thnhan
"""

import pandas as pd
import numpy as np

from data.fasta_tool import fasta_to_dataframe


def load_raw_dset(dset_dir):
    print("\n... Load from " + dset_dir + "/uniprotein.csv")

    seq = pd.read_csv(dset_dir + '/uniprotein.csv', index_col=0)
    pos = pd.read_csv(dset_dir + '/pairs_pos.csv')
    neg = pd.read_csv(dset_dir + '/pairs_neg.csv')

    do_dai = sorted([len(p) for p in seq.protein])
    avelen = sum(do_dai) / len(do_dai)
    summary = {'minlen': do_dai[0], 'maxlen': do_dai[-1], 'avelen': avelen, 'n_proteins': len(do_dai)}

    P_seq_A = seq.loc[pos.proteinA]['protein'].values
    P_seq_B = seq.loc[pos.proteinB]['protein'].values
    N_seq_A = seq.loc[neg.proteinA]['protein'].values
    N_seq_B = seq.loc[neg.proteinB]['protein'].values

    labels = np.array([1] * len(pos) + [0] * len(neg))
    pairs = np.vstack((pos.values, neg.values))
    dset = {"labels": labels, "id_pairs": pairs, "seq_pairs": (P_seq_A, P_seq_B, N_seq_A, N_seq_B)}
    return dset, summary


def load_Yeastfull_new(dset_dir):
    seq = fasta_to_dataframe(dset_dir + '/uniprotein.csv')
    P_seq_A = fasta_to_dataframe(dset_dir + '/pos_A.txt')
    P_seq_B = fasta_to_dataframe(dset_dir + '/pos_B.txt')
    N_seq_A = fasta_to_dataframe(dset_dir + '/neg_A.txt')
    N_seq_B = fasta_to_dataframe(dset_dir + '/neg_B.txt')

    do_dai = [len(p) for p in seq.protein]
    avelen = sum(do_dai) / len(do_dai)
    summary = {'minlen': do_dai[0], 'maxlen': do_dai[-1], 'avelen': avelen, 'n_proteins': len(do_dai)}

    yeastfull = pd.read_csv(dset_dir + "/yeastfull_pair.txt")
    labels = yeastfull['interaction'].values
    pairs = yeastfull[['proteinA', 'proteinB']]
    dset = {"labels": labels, "id_pairs": pairs, "uniprot": seq, "seq_pairs": (P_seq_A, P_seq_B, N_seq_A, N_seq_B)}
    return dset, summary


if __name__ == "__main__":
    dset_, summary_ = load_raw_dset(r"Human")
    print("summary:", summary_)
    print(dset_.keys())
    P_seq_A_, P_seq_B_, N_seq_A_, N_seq_B_ = dset_['seq_pairs']
    print("pair\n", P_seq_A_)
