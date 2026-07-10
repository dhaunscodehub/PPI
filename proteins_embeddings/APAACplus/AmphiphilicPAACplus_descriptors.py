"""
Plus v5

http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/type2.htm

Type 2 PseAA composition is also called the series-correlation type and generates 20 + i*lambda
discrete numbers to represent a protein (i is the number of amino acid attributes selected),
which was introduced by Prof. Kuo-Chen Chou in 2005 and the related publications are:

(1) Chou, K.C. (2005). Using amphiphilic pseudo amino acid composition to predict enzyme subfamily classes, Bioinformatics, 21, 10-19.
(2) Chou,K.C. and Cai Y.D. (2005). Prediction of membrane protein types by incorporating amphipathic effects, J Chem Inf Model, 45(2):407-13
"""

import numpy as np
import pandas as pd

from protein_descriptors.APAACplus.AAC import AACEncoder


def read_supp_table(supp):
    table = pd.read_csv(supp)
    table.set_index('AA', inplace=True)
    return table


def to_indices(sequence_, dict_AA_1):
    return [dict_AA_1[aa] for aa in sequence_]


def tau_di_peptides(supp_table, AA_1):
    tau1 = np.zeros((20, 20))
    tau2 = np.zeros((20, 20))

    proper1 = supp_table['H1'].tolist()
    proper2 = supp_table['H2'].tolist()

    # tau1, tau2
    for i, aa1 in enumerate(AA_1):
        for j, aa2 in enumerate(AA_1[i:]):
            v1 = proper1[i] * proper1[j]
            tau1[i][j] = v1
            tau1[j][i] = v1
            v2 = proper2[i] * proper2[j]
            tau2[i][j] = v2
            tau2[j][i] = v2

    return tau1, tau2


def tau_v2_tripeptides(supp_table, sequence_, lag):
    proper1 = supp_table['H1'].tolist()
    proper2 = supp_table['H2'].tolist()

    # global tau1_v2, tau2_v2
    temp1, temp2 = 0.0, 0.0
    # print(sequence_)
    for i, aa1 in enumerate(sequence_):
        j = i + lag
        k = j + lag
        if k >= len(sequence_):
            break

        id_i = sequence_[i]
        id_j = sequence_[j]
        id_k = sequence_[k]

        # v1 = (proper1[id_i] + proper1[id_k]) * (proper1[id_j] + proper1[id_k])
        # temp1 += v1
        # v2 = (proper2[id_i] + proper2[id_k]) * (proper2[id_j] + proper2[id_k])
        # temp2 += v2

        v1 = proper1[id_i] * proper1[id_j] * proper1[id_k]
        temp1 += v1
        v2 = proper2[id_i] * proper2[id_j] * proper2[id_k]
        temp2 += v2

    return temp1, temp2


class APAACEncoderPlus:
    def __init__(self, lg=30, supp=None, path_to_supp_file="protein_descriptors/APAACplus/supp/supp_APAAC.csv"):
        if supp is not None:
            self.supp_table = read_supp_table(supp)
        else:
            self.supp_table = read_supp_table(path_to_supp_file)
        self.shortName = 'APAAC Plus'
        self.fullName = 'Amphiphilic Pseudo Amino Acid Composition Plus'
        self.AA_1 = "ARNDCEQGHILKMFPSTWYV"
        self.AA_idx = {'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'E': 5, 'Q': 6, 'G': 7, 'H': 8, 'I': 9, 'L': 10, 'K': 11,
                       'M': 12, 'F': 13, 'P': 14, 'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19}
        self.lg = lg
        self.tau1, self.tau2 = tau_di_peptides(self.supp_table, self.AA_1)

    # @staticmethod
    def to_feature(self, sequence_, lg=30, w1=0.5, w2=0.5):
        lg = self.lg

        # Thành phần 1
        F = AACEncoder().to_feature(sequence_)

        # Thành phần 2
        tau_di_peptides(self.supp_table, self.AA_1)
        # tau_v2_di_peptides(self.supp_table)
        seq = to_indices(sequence_, self.AA_idx)

        tau = [0.0] * (2 * lg)
        tau_v2 = [0.0] * (2 * lg)

        L = len(sequence_)
        i = 0
        for k in range(1, lg + 1):
            if k >= L:
                s1 = 0.
                s2 = 0.
            else:
                s1 = sum([self.tau1[seq[j]][seq[j + k]] for j in range(L - k)])
                s2 = sum([self.tau2[seq[j]][seq[j + k]] for j in range(L - k)])
            tau[i] = (0.5 * s1) / (len(sequence_) - k + 0.000001)
            tau[i + 1] = (0.5 * s2) / (len(sequence_) - k + 0.000001)
            i += 2

        i = 0
        for k in range(1, lg + 1):
            # new
            if 2*k >= L:
                s1_v2 = 0.
                s2_v2 = 0.
            else:
                s1_v2, s2_v2 = tau_v2_tripeptides(self.supp_table, seq, k)
            # print(k, s1_v2, s2_v2)
            # print(len(sequence_))
            tau_v2[i] = (0.5 * s1_v2) / (len(sequence_) - 2 * k + 0.000001)
            tau_v2[i + 1] = (0.5 * s2_v2) / (len(sequence_) - 2 * k + 0.000001)
            i += 2

        mau = 1.0 + w1 * sum(tau) + w2 * sum(tau_v2)
        # mau = 1.0 + 0.5 * sum(tau) + 0.5 * sum(tau_v2)
        features = np.hstack((F, tau, tau_v2))
        return features / mau


if __name__ == "__main__":
    # sequence = 'AFQVNTNINAMNAHVQSALTQNALKTSLERLSSGLRINKAADDASGMTVADSLRSQASSLGQAIANTNDGMGIIQVADKAMDEQLKILDTVKVKAT' \
    #            'QAAQDGQTTESRKAIQSDIVRLIQGLDNIGNTTTYNGQALLSGQFTNKEFQVGAYSNQSIKASIGSTTSDKIGQVRIATGALITASGDISLTFKQV' \
    #            'DGVNDVTLESVKVSSSAGTGIGVLAEVINKNSNRTGVKAYASVITTSDVAVQSGSLSNLTLNGIHLGNIADIKKNDSDGRLVAAINAVTSETGVEA' \
    #            'YTDQKGRLNLRSIDGRGIEIKTDSVSNGPSALTMVNGGQDLTKGSTNYGRLSLTRLDAKSINVVSASDSQHLGFTAIGFGESQVAETTVNLRDVTG' \
    #            'NFNANVKSASGANYNAVIASGNQSLGSGVTTLRGAMVVIDIAESAMKMLDKVRSDLGSVQNQMISTVNNISITQVNVKAAESQIRDVDFAEESANF' \
    #            'NKNNILAQSGSYAMSQANTVQQNILRLLT'
    sequence = 'AFQVNTNINAMNAHV'
    feat = APAACEncoderPlus(path_to_supp_file="supp/supp_APAAC.csv").to_feature(sequence)
    print(feat)
    print(feat.shape)
