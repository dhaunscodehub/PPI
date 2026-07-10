import argparse
from sys import path
from collections import Counter
import numpy as np

from protein_descriptors.APAACplus.AA import AA_1

path.append(path[0] + '/..')


class AACEncoder:
    def __init__(self):
        self.minLength = 1  # Length conditions
        self.dim = 20
        self.shortName = 'AAC'
        self.fullName = 'Amino Acid Composition'

    @staticmethod
    def to_feature(sequence):
        n_AA = Counter(sequence)
        L = len(sequence)
        return np.array([n_AA[aa] / L for aa in AA_1])

    def example(self, sequence=None):
        if sequence is None:
            sequence \
                = 'AFQVNTNINAMNAHVQSALTQNALKTSLERLSSGLRINKAADDASGMTVADSLRSQASSLGQAIANTNDGMGIIQVADKAMDEQLKILDTVKVKA' \
                  'TQAAQDGQTTESRKAIQSDIVRLIQGLDNIGNTTTYNGQALLSGQFTNKEFQVGAYSNQSIKASIGSTTSDKIGQVRIATGALITASGDISLTFK' \
                  'QVDGVNDVTLESVKVSSSAGTGIGVLAEVINKNSNRTGVKAYASVITTSDVAVQSGSLSNLTLNGIHLGNIADIKKNDSDGRLVAAINAVTSETG' \
                  'VEAYTDQKGRLNLRSIDGRGIEIKTDSVSNGPSALTMVNGGQDLTKGSTNYGRLSLTRLDAKSINVVSASDSQHLGFTAIGFGESQVAETTVNLR' \
                  'DVTGNFNANVKSASGANYNAVIASGNQSLGSGVTTLRGAMVVIDIAESAMKMLDKVRSDLGSVQNQMISTVNNISITQVNVKAAESQIRDVDFAE' \
                  'ESANFNKNNILAQSGSYAMSQANTVQQNILRLLT'
        print('Vector Dimension', self.dim)
        print('Features of protein "{}"\n{}'.format(
            sequence,
            self.to_feature(sequence)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastProtFeat.")

    parser.add_argument("--path",
                        nargs="?",
                        default="example",
                        help="Path of protein dataset")

    parser.add_argument("--descriptors",
                        nargs="?",
                        default="AAC",
                        help="The name of descriptors, which will be to convert proteins to nummerical features")

    # if parser.parse_args().descriptors == 'AAC':
    parser.add_argument("--param1",
                        type=int,
                        default=1,
                        help="parameter 1")
    args = parser.parse_args()
    seq = 'AFQVNTNINAMNAHVQSALTQNALKTSLERLSSGLRINKAADDASGMTVADSLRSQASSLGQAIANTNDGMGIIQVADKAMDEQLKILDTVKVKA' \
          'TQAAQDGQTTESRKAIQSDIVRLIQGLDNIGNTTTYNGQALLSGQFTNKEFQVGAYSNQSIKASIGSTTSDKIGQVRIATGALITASGDISLTFK' \
          'QVDGVNDVTLESVKVSSSAGTGIGVLAEVINKNSNRTGVKAYASVITTSDVAVQSGSLSNLTLNGIHLGNIADIKKNDSDGRLVAAINAVTSETG' \
          'VEAYTDQKGRLNLRSIDGRGIEIKTDSVSNGPSALTMVNGGQDLTKGSTNYGRLSLTRLDAKSINVVSASDSQHLGFTAIGFGESQVAETTVNLR' \
          'DVTGNFNANVKSASGANYNAVIASGNQSLGSGVTTLRGAMVVIDIAESAMKMLDKVRSDLGSVQNQMISTVNNISITQVNVKAAESQIRDVDFAE' \
          'ESANFNKNNILAQSGSYAMSQANTVQQNILRLLT'
    # print('PATH', args.path)
    # print('RESULTS')
    print(AACEncoder().to_feature(seq))
