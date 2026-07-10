"""
@coding: thnhan
"""
from collections import Counter
from numpy import array, zeros

# #########################################################################
# Các nhóm theo thứ tự của bài LightGBM-PPI,
# doi: https://doi.org/10.1016/j.chemolab.2019.06.003.
# #########################################################################
LightGBM_G_idx = {
    'A': 1, 'G': 1, 'V': 1,
    'I': 2, 'L': 2, 'F': 2, 'P': 2,
    'Y': 3, 'M': 3, 'T': 3, 'S': 3, 'U': 3, 'X': 3,
    'H': 5, 'N': 5, 'Q': 5, 'W': 5,
    'R': 4, 'K': 4,
    'D': 6, 'E': 6,
    'C': 7,
    # 'X': '6', 'B': '6', 'U': '6'  # Nhung amino acid không xác định đc co vào nhóm 6 theo LR_PPI
}


def to_group_idx(sequence, AA_G_idx=None):
    """ AA_G_idx là bảng ánh xạ từ AA và chỉ số nhóm """
    if AA_G_idx is None:
        AA_G_idx = LightGBM_G_idx
    return [AA_G_idx[aa] for aa in sequence]


def to_CTriad_old(sequence, triplets, gap=0):
    L = len(sequence) - 3 - 2 * gap
    seq = [(sequence[i], sequence[i + gap + 1], sequence[i + 2 * gap + 2]) for i in range(L)]
    count = Counter(seq)
    min_f_i = list(count.values())[-1]
    max_f_i = list(count.values())[0]

    return [(count[p] - min_f_i) / max_f_i for p in triplets]


def to_CTriad(sequence, gap=0):
    L = len(sequence) - 3 - 2 * gap
    count = [[[0] * 7] * 7] * 7
    for i in range(L):
        # print(f'{sequence[i]}, {sequence[i+1]}, {sequence[i+2]}')
        count[sequence[i] - 1][sequence[i + gap + 1] - 1][sequence[i + 2 * gap + 2] - 1] += 1
    count = array(count).reshape(-1)
    return (count - count.min()) / count.max()


class KSCTriadEncoder:
    def __init__(self, k=5):
        # super().__init__()
        self.minLength = 3 + 2 * (k + 1)  # Length conditions
        self.dim = 373 * (k + 1)
        self.shortName = 'k-SCTriad'
        self.fullName = 'k-Spaced Conjoint Triad'

    def to_feature(self, sequence, k=5):
        if len(sequence) < self.minLength:
            print("Error length")
            return None

        # print(k)
        # print(self.dim)

        # Chuyển AA trong sequence thành chỉ số nhóm
        sequence = to_group_idx(sequence)
        # print(sequence)

        # # Ghép bộ ba
        # triplets = [(i, j, k) for i in range(7) for j in range(7) for k in range(7)]

        # # Tạo vector đặc trưng
        # feature_extraction = []
        # for gap in range(k + 1):
        #     feature_extraction.append(to_CTriad(sequence, triplets, gap))
        # return array(feature_extraction).reshape(-1)

        # Tạo vector đặc trưng
        features = [0.0] * self.dim
        i = 0
        for gap in range(k + 1):
            features[i:i + 343] = to_CTriad(sequence, gap)
            i += 343
        return features

    # def example(self, sequence=None):
    #     if sequence is None:
    #         sequence \
    #             = 'AFQVNTNINAMNAHVQSALTQNALKTSLERLSSGLRINKAADDASGMTVADSLRSQASSLGQAIANTNDGMGIIQVADKAMDEQLKILDTVKVKA' \
    #               'TQAAQDGQTTESRKAIQSDIVRLIQGLDNIGNTTTYNGQALLSGQFTNKEFQVGAYSNQSIKASIGSTTSDKIGQVRIATGALITASGDISLTFK' \
    #               'QVDGVNDVTLESVKVSSSAGTGIGVLAEVINKNSNRTGVKAYASVITTSDVAVQSGSLSNLTLNGIHLGNIADIKKNDSDGRLVAAINAVTSETG' \
    #               'VEAYTDQKGRLNLRSIDGRGIEIKTDSVSNGPSALTMVNGGQDLTKGSTNYGRLSLTRLDAKSINVVSASDSQHLGFTAIGFGESQVAETTVNLR' \
    #               'DVTGNFNANVKSASGANYNAVIASGNQSLGSGVTTLRGAMVVIDIAESAMKMLDKVRSDLGSVQNQMISTVNNISITQVNVKAAESQIRDVDFAE' \
    #               'ESANFNKNNILAQSGSYAMSQANTVQQNILRLLT'
    #     vec = self.to_feature(sequence)
    #     print('Protein:', sequence)
    #     print('Features:')
    #     print(vec)
    #     print('Dimension', len(vec))


if __name__ == "__main__":
    sequence \
        = 'AFQVNTNINAMNAHVQSALTQNALKTSLERLSSGLRINKAADDASGMTVADSLRSQASSLGQAIANTNDGMGIIQVADKAMDEQLKILDTVKVKA' \
          'TQAAQDGQTTESRKAIQSDIVRLIQGLDNIGNTTTYNGQALLSGQFTNKEFQVGAYSNQSIKASIGSTTSDKIGQVRIATGALITASGDISLTFK' \
          'QVDGVNDVTLESVKVSSSAGTGIGVLAEVINKNSNRTGVKAYASVITTSDVAVQSGSLSNLTLNGIHLGNIADIKKNDSDGRLVAAINAVTSETG' \
          'VEAYTDQKGRLNLRSIDGRGIEIKTDSVSNGPSALTMVNGGQDLTKGSTNYGRLSLTRLDAKSINVVSASDSQHLGFTAIGFGESQVAETTVNLR' \
          'DVTGNFNANVKSASGANYNAVIASGNQSLGSGVTTLRGAMVVIDIAESAMKMLDKVRSDLGSVQNQMISTVNNISITQVNVKAAESQIRDVDFAE' \
          'ESANFNKNNILAQSGSYAMSQANTVQQNILRLLT'
    v = KSCTriadEncoder().to_feature(sequence)
    print(v)
    print(len(v))
