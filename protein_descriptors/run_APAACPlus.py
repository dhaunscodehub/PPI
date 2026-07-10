import pickle

import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat

from protein_descriptors.APAACplus.AmphiphilicPAACplus_descriptors import APAACEncoderPlus
from utils.read_mat_file import loadmat_TEST_seq, loadmat_TEST_feat, loadmat_feat_to_dataframe


def prot_to_feat(protein, all_enc, remove_unknown_AA=True):
    # Loại bỏ các axit amin không xác định 'U', 'train_X'
    if remove_unknown_AA:
        # protein = re.sub("[UX]", "Z", protein)
        protein = protein.replace("U", "")
        protein = protein.replace("X", "")

    feat = [enc.to_feature(protein) for enc in all_enc]
    # handfeat = np.concatenate(handfeat)
    # f1 = all_enc[0].to_feature(protein)
    # f2 = all_enc[1].to_feature(protein)
    return np.concatenate(feat)


def prot_list_to_feat(proteins, all_enc, remove_unknown_AA=False):
    """ @thnhan
    1. Lọc ra các trình tự protein khác nhau
    2. Chuyển trình tự protein thành vector đặc trưng
    3. Lắp ghếp các vector đặc trưng theo đúng thứ tự các chuỗi trong proteins
    """

    # if len(proteins) >= 500:
    #     uni_prots = pd.unique(proteins)  # uni_prots chứa những protein duy nhất
    #     uni_feats = map(lambda prot: prot_to_feat(prot, all_enc, remove_unknown_AA), uni_prots)
    #     # print(list(uni_feats))
    #     feat_list = pd.DataFrame(data=list(uni_feats), index=uni_prots)
    #     feat_list = feat_list.loc[proteins]
    #     return feat_list.values
    # else:
    if type(proteins) != list:
        proteins = proteins.protein.values.flatten()

    feat_list = map(lambda prot: prot_to_feat(prot, all_enc), proteins)
    return np.array(list(feat_list))


def convert(proteins, protein_encoders):
    print("Number of proteins:", len(proteins))
    all_feat = prot_list_to_feat(proteins,
                                 protein_encoders,
                                 remove_unknown_AA=True)
    # print("Done.")
    return all_feat


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Run on Human dataset
    # -----------------------------------------------------------------
    lg = 30
    uniprot = pd.read_csv("../data/Human/uniprotein.csv", sep="\t")
    protID, uniseq = uniprot.id, uniprot.protein

    print("\nlg", lg)
    protein_encoders = [
        APAACEncoderPlus(lg=lg)
    ]

    unifeat = convert(uniprot, protein_encoders)

    protID = np.array(protID.values.reshape(-1, 1), dtype='object')
    savemat(file_name='uni_APAACPlus_Human_lg' + str(lg) + '.mat',
            mdict={'prot_id': protID, 'prot_feat': unifeat})

    data = loadmat_feat_to_dataframe('uni_APAACPlus_Human_lg' + str(lg) + '.mat')

    print(data)

    # # -----------------------------------------------------------------
    # # Run on TEST dataset
    # # -----------------------------------------------------------------
    # names = [
    #     # 'Cancer', 'Celeg', 'Dmela', 'Ecoli', 'Hpylo', 'Hsapi', 'Mmusc', 'MultiCore',
    #     'Onecore', 'Wnt'
    # ]
    # for name in names:
    #     prot_A, prot_B = loadmat_TEST_seq('../data/Testsets/' + name + '.mat')
    #     print('\nRunning feature extraction ...', len(prot_B))
    #
    #     protein_encoders = [
    #         APAACEncoderPlus(path_to_supp_file="APAACplus/supp/supp_APAAC.csv")
    #     ]
    #
    #     feat_A = convert(prot_A, protein_encoders)
    #     feat_B = convert(prot_B, protein_encoders)
    #     print('done protein A', feat_A.shape)
    #     print('done protein B', feat_B.shape)
    #
    #     savemat(file_name='feat_' + name + '_APAACPlus_lg30.mat', mdict={'Pa': feat_A, 'Pb': feat_B})
    #
    #     # Try to read
    #     feat_A, feat_B = loadmat_TEST_feat('feat_' + name + '_APAACPlus_lg30.mat')
    #     print(feat_A.shape)
