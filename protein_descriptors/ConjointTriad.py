"""
@coding: thnhan
"""
from protein_descriptors.KSConjointTriad import KSCTriadEncoder


# from KSConjointTriad import KSCTriadEncoder


class ConjointTriadEncoder:
    def __init__(self):
        # super().__init__()
        self.minLength = 3  # Length conditions
        self.dim = 373
        self.shortName = 'CTriad'
        self.fullName = 'Conjoint Triad'

    @staticmethod
    def to_feature(sequence):
        return KSCTriadEncoder(k=0).to_feature(sequence, k=0)
        # if len(sequence) >= self.minLength:
        #     print("Error length")
        #     return -1
        #
        # if sequence[0] not in ['1', '2', '3', '4', '5', '6', '7']:
        #     sequence = to_group_idx(sequence)
        #
        # n_components = count_TC(sequence)
        # feature = np.array(list(n_components.values()))
        # min_f_i = np.min(feature)
        # max_f_i = np.max(feature)
        # feature = (feature - min_f_i) / max_f_i
        #
        # return feature

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
    # prot = 'MAAAAAAAAATNGTGGSSGMEVDAAVVPSVMACGVTGSVSVALHPLVILNISDHWIRMRSQEGRPVQVIGALIGKQEGRNIEVMNSFELLSHTVEEKIIIDKEY' \
    #        'YYTKEEQFKQVFKELEFLGWYTTGGPPDPSDIHVHKQVCEIIESPLFLKLNPMTKHTDLPVSVFESVIDIINGEATMLFAELTYTLATEEAERIGVDHVARMTA' \
    #        'TGSGENSTVAEHLIAQHSAIKMLHSRVKLILEYVKASEAGEVPFNHEILREAYALCHCLPVLSTDKFKTDFYDQCNDVGLMAYLGTITKTCNTMNQFVNKFNVL' \
    #        'YDRQGIGRRMRGLFF'
    prot = 'MEGVYFNIDNGFIEGVVRGYRNGLLSNNQYINLTQCDTLEDLKLQLSSTDYGNFLSSVSSESLTTSLIQEYASSKLYHEFNYIRDQSSGSTRKFMDYITYGYMIDNVALMITGTIHDRDKGEILQRCHPLGWFDTLPTLSVATDLESLYETVLVDTPLAPYFKNCFDTAEELDDMNIEIIRNKLYKAYLEDFYNFVTEEIPEPAKECMQTLLGFEADRRSINIALNSLQSSDIDPDLKSDLLPNIGKLYPLATFHLAQAQDFEGVRAALANVYEYRGFLETGNLEDHFYQLEMELCRDAFTQQFAISTVWAWMKSKEQEVRNITWIAECIAQNQRERINNYISVY'
    # ConjointTriadEncoder().example()
    v = ConjointTriadEncoder().to_feature(prot)
    print(len(v))
    print(v)
