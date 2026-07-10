from itertools import combinations
import math
from collections import Counter
import numpy as np

group = [
    ['A', 'G', 'V'],
    ['C'],
    ['D', 'E'],
    ['F', 'P', 'I', 'L'],
    ['H', 'Q', 'N', 'W'],
    ['K', 'R'],
    ['M', 'S', 'Y', 'T']
]


def aa2group(aa, in_group=None):
    """ aa: amino acid """
    if in_group is None:
        in_group = group

    ii = 0
    while (aa not in in_group[ii]) and (ii < 7):
        ii += 1

    if ii == 7:
        ii = -1
    return ii


def sequence2group(seq, in_group=None):
    return [aa2group(aa_, in_group) for aa_ in seq]


def to_feature(sequence, ggg, gg, t, n_groups=35):
    f_vector = []
    for i in range(n_groups):
        cc = list(gg.difference(set(ggg[i])))
        # print('cc', list(cc))

        c_aa = sequence2group(sequence)
        tmp = []
        for aa in c_aa:
            tam = 2
            while cc[tam] != aa and tam > 0:
                tam -= 1
            tmp.append(t[tam])
        # print(tmp)

        counts = dict()
        for j in tmp:
            counts[j] = counts.get(j, 0) + 1

        # print(counts)
        nB = 0
        nJ = 0
        nO = 0
        nU = 0
        x, y = [], []
        pi_2 = math.pi / 2
        for aa in tmp:
            if aa == 'B':
                nB += 1
                x.append(math.cos(pi_2 * nB / counts['B']))
                y.append(math.sin(pi_2 * nB / counts['B']))
            elif aa == 'J':
                nJ += 1
                x.append(math.cos(pi_2 + pi_2 * nJ / counts['J']))
                y.append(math.sin(pi_2 + pi_2 * nJ / counts['J']))
            elif aa == 'O':
                nO += 1
                x.append(math.cos(2 * pi_2 + pi_2 * nO / counts['O']))
                y.append(math.sin(2 * pi_2 + pi_2 * nO / counts['O']))
            elif aa == 'U':
                nU += 1
                x.append(math.cos(3 * pi_2 + pi_2 * nU / counts['U']))
                y.append(math.sin(3 * pi_2 + pi_2 * nU / counts['U']))

        # print(x)
        # print(y)

        n = len(x)
        mx = sum(x) / len(x)
        my = sum(y) / len(y)
        vx = np.sqrt(np.sum(np.subtract(x, mx) ** 2) / (n - 1))
        vy = np.sqrt(np.sum(np.subtract(y, my) ** 2) / (n - 1))
        f_vector.append([mx, my, vx, vy])

    f_vector = np.array(f_vector).flatten()
    return f_vector


class Fvector:
    def __init__(self, n_groups=35):
        self.gg = [0, 1, 2, 3, 4, 5, 6]
        self.ggg = list(combinations(self.gg, 4))
        self.gg = set(self.gg)
        self.t = {0: 'B', 1: 'J', 2: 'O', 3: 'U'}
        self.n_groups = n_groups

    def to_feature(self, seq):
        return to_feature(seq, self.ggg, self.gg, self.t, self.n_groups)


if __name__ == "__main__":
    prot_seq = 'KRSAGHQ'

    v = Fvector().to_feature(prot_seq)
    print(v)
    print(v.shape)
