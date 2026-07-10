"""
http://www.csbio.sjtu.edu.cn/bioinf/PseAAC/type2.htm

Type 2 PseAA composition is also called the series-correlation type and generates 20 + i*lambda
discrete numbers to represent a protein (i is the number of amino acid attributes selected),
which was introduced by Prof. Kuo-Chen Chou in 2005 and the related publications are:

(1) Chou, K.C. (2005). Using amphiphilic pseudo amino acid composition to predict enzyme subfamily classes, Bioinformatics, 21, 10-19.
(2) Chou,K.C. and Cai Y.D. (2005). Prediction of membrane protein types by incorporating amphipathic effects, J Chem Inf Model, 45(2):407-13
"""
# from FastProtFeat.AA import AA_1, AA_idx
# from FastProtFeat.runAAC import AACEncoder

import numpy as np
import pandas as pd

from new_descriptors_for_bai_moi.APAAC.AA import AA_1, AA_idx
from new_descriptors_for_bai_moi.APAAC.AAC import AACEncoder


def read_supp_table(supp):
    table = pd.read_csv(supp)
    table.set_index('AA', inplace=True)
    return table


def to_indices(sequence, dict_AA_1):
    return [dict_AA_1[aa] for aa in sequence]


tau1 = [[0.0] * 20] * 20
tau2 = [[0.0] * 20] * 20


def tau_di_peptides(supp_table):
    proper1 = supp_table['H1'].tolist()
    proper2 = supp_table['H2'].tolist()

    global tau1, tau2
    for i, aa1 in enumerate(AA_1):
        for j, aa2 in enumerate(AA_1[i:]):
            v1 = proper1[i] * proper1[j]
            tau1[i][j] = v1
            tau1[j][i] = v1
            v2 = proper2[i] * proper2[j]
            tau2[i][j] = v2
            tau2[j][i] = v2


class APAACEncoder:
    def __init__(self, lg=30, supp=None, path_to_supp_file="new_descriptors_for_bai_moi/APAAC/supp/supp_APAAC.csv"):
        if supp is not None:
            self.supp_table = read_supp_table(supp)
        else:
            self.supp_table = read_supp_table(path_to_supp_file)
        self.minLength = 31
        self.dim = 20 + 2 * lg
        self.shortName = 'APAAC'
        self.fullName = 'Amphiphilic Pseudo Amino Acid Composition'

    # @staticmethod
    def to_feature(self, sequence_, lg=30, omega=0.5):
        # Thành phần 1
        F = AACEncoder().to_feature(sequence_)

        # Thành phần 2
        tau_di_peptides(self.supp_table)
        seq = to_indices(sequence_, AA_idx)
        tau = [0.0] * (2 * lg)
        L = len(sequence_)
        i = 0
        for k in range(1, lg + 1):
            s1 = sum([tau1[seq[i]][seq[i + k]] for i in range(L - k)])
            s2 = sum([tau2[seq[i]][seq[i + k]] for i in range(L - k)])
            tau[i] = (omega * s1) / (len(sequence_) - k + 0.000001)
            tau[i + 1] = (omega * s2) / (len(sequence_) - k + 0.000001)
            i += 2

        mau = 1.0 + omega * sum(tau)
        features = np.hstack((F, tau))

        return features / mau

    def example(self, sequence_=None):
        if sequence_ in None:
            sequence_ = 'AFQVNTNINAMNAHVQSALTQNALKTSLERLSSGLRINKAADDASGMTVADSLRSQASSLGQAIANTNDGMGIIQVADKAMDEQLKILDTVKVKAT' \
                        'QAAQDGQTTESRKAIQSDIVRLIQGLDNIGNTTTYNGQALLSGQFTNKEFQVGAYSNQSIKASIGSTTSDKIGQVRIATGALITASGDISLTFKQV' \
                        'DGVNDVTLESVKVSSSAGTGIGVLAEVINKNSNRTGVKAYASVITTSDVAVQSGSLSNLTLNGIHLGNIADIKKNDSDGRLVAAINAVTSETGVEA' \
                        'YTDQKGRLNLRSIDGRGIEIKTDSVSNGPSALTMVNGGQDLTKGSTNYGRLSLTRLDAKSINVVSASDSQHLGFTAIGFGESQVAETTVNLRDVTG' \
                        'NFNANVKSASGANYNAVIASGNQSLGSGVTTLRGAMVVIDIAESAMKMLDKVRSDLGSVQNQMISTVNNISITQVNVKAAESQIRDVDFAEESANF' \
                        'NKNNILAQSGSYAMSQANTVQQNILRLLT'
        print('Features Dimensional', self.dim)
        print(f'Features {sequence_}\n{self.to_feature(sequence_)}')


if __name__ == "__main__":
    sequence = 'AFQVNTNINAMNAHVQSALTQNALKTSLERLSSGLRINKAADDASGMTVADSLRSQASSLGQAIANTNDGMGIIQVADKAMDEQLKILDTVKVKAT'
    feat = APAACEncoder().to_feature(sequence)
    print(feat)
    print(feat.shape)
