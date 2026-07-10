""" code: thnhan """
import csv
import pickle

import numpy as np


def to_kgrams_overlap(sequence, k_grams, overlap=2):
    if k_grams <= overlap:
        return "ERROR"
    n = len(sequence) - k_grams + 1
    new = [sequence[ii:ii + k_grams] for ii in range(0, n, k_grams - overlap)]
    # print(new)
    return new


def to_kgrams(sequence: str, k_grams: int = 1, overlap: int = 0):
    if k_grams < 1:
        return "ERROR type"

    if k_grams == 1:
        return list(sequence)

    return to_kgrams_overlap(sequence, k_grams, overlap)


def prot_to_token_id(vocal, embeddings, protseqs, prot_len, k_gram=3):
    def to_token(prot, n_vocabs=len(vocal)):
        protkn = []
        if len(prot) >= prot_len:
            prot3mers = [prot[ii:ii + 3] for ii in range(0, prot_len - prot_len % 3, 3)]
            if prot_len % 3 > 0:
                prot3mers.append('<unk>')

            for kmer_ in prot3mers:
                if kmer_ not in vocal:
                    print(kmer_, "not in vocal")
                    # exit(1)
                    # vocal[kmer_] = len(vocal) - 1
                    protkn.append(vocal['<unk>'])
                else:
                    protkn.append(vocal[kmer_])

            # if len(protkn) == 184:
            #     print("protkn", protkn)
        else:
            # print(len(prot))
            prot3mers = [prot[ii:ii + 3] for ii in range(0, len(prot) - len(prot) % 3, 3)]
            if prot_len % 3 > 0:
                prot3mers.append('<unk>')
            # print("prot3mers", len(prot3mers))
            for kmer_ in prot3mers:
                if kmer_ not in vocal:
                    print("kmer_ not in vocal", kmer_)
                    # exit(1)
                    # vocal[kmer_] = len(vocal) - 1
                    protkn.append(vocal['<unk>'])
                else:
                    protkn.append(vocal[kmer_])
            # print("protkn", len(protkn))
            # print("protkn", protkn)
            # print('con lai', prot_len//3 - len(protkn))
            nhucc = prot_len // 3 - len(protkn)
            if prot_len % 3 > 0:
                nhucc += 1
            for _ in range(nhucc):
                protkn.append(vocal['<unk>'])
            if len(protkn) == 184:
                print("protkn", protkn)
        return protkn

    vocal['_'] = len(vocal) - 1  # them ky tu "_" lam pad
    tokens = np.array(list(map(to_token, protseqs)))

    print(tokens)

    # all_tokens = []
    # for protseq in protseqs:
    #     prot3mers = [protseq[ii:ii + 3] for ii in range(0, len(protseq) - 2, 3)]
    #     if len(protseq) % 3 > 0:
    #         prot3mers.append('<unk>')
    #     # print("prot3mers", prot3mers)
    #
    #     protkn = np.array([vocal[kmer_] for kmer_ in prot3mers])
    #     # print("protkn", protkn)
    #
    #     all_tokens.append(protkn)
    # print(len(tokens[0]))
    return tokens


def tokenize_id_list_protein(list_protein, vocabulary, prot_len, k_gram=3):
    all_token_id = []
    for prot_seq in list_protein:
        # --- Fix legnth
        if len(prot_seq) >= prot_len:
            fix_prot_seq = prot_seq[:prot_len]
        else:
            fix_prot_seq = prot_seq + '_' * (prot_len - len(prot_seq))
        # print(fix_prot_seq)

        token = to_kgrams(fix_prot_seq, k_gram, 0)
        # print(token)

        # --- to token_id
        token_id = []
        for t in token:
            id_ = vocabulary.get(t, vocabulary['<unk>'])

            token_id.append(id_)
        all_token_id.append(token_id)
    return np.array(all_token_id)


def prepare_YeastCore_Doc2vec_token(vocabulary, W1, protlen, dset, k_gram):
    pos_seq_A, pos_seq_B, neg_seq_A, neg_seq_B = dset['seq_pairs']

    seqs_A = np.concatenate([pos_seq_A, neg_seq_A], axis=0)
    seqs_B = np.concatenate([pos_seq_B, neg_seq_B], axis=0)

    # tokens_A = tokenize_id_list_protein(seqs_A, vocabulary, protlen, k_gram=k_gram)  # new
    # tokens_B = tokenize_id_list_protein(seqs_B, vocabulary, protlen, k_gram=k_gram)  # new

    tokens_A = prot_to_token_id(vocabulary, W1, seqs_A, protlen)  # old
    tokens_B = prot_to_token_id(vocabulary, W1, seqs_B, protlen)  # old

    return tokens_A, tokens_B


if __name__ == '__main__':
    proteins = [
        "GRVIRNQRKGAGSIFTSHTRLRQGAAKLRTLDYAERHGYIRGIVKQIVHDSGRGAPLAKVVFRDPYKYRLREEIFIANEGVHTGQFIYAGKKASLNVGNVLP",
        "LGSVPEGTIVSNVEEKPGDRGALARASGNYVIIIGHNPDENKTRVRLPSGAKKVISSDARGVIGVIAGGGRVDKPLLKAGRAFHKYRLKRNSWPKTRGVAMN",
        "PVDHPHGGGNHQHIGKASTISRGAVSG"]

    data = pickle.load(open("d2v_3gram_32dim.pkl", "rb"))

    # print(data['vocabulary'])

    data['vocabulary']['<unk>'] = len(data['vocabulary'])
    print("Embedding", data['W1'].shape)
    print("Vocabulary", len(data['vocabulary']))

    list_tokenid_1 =tokenize_id_list_protein(proteins, data['vocabulary'], 60, k_gram=3)
    print(list_tokenid_1)

    print("\n")

    list_tokenid_2 = prot_to_token_id(data['vocabulary'], data['W1'], proteins, 60, 3)
    print(list_tokenid_2)

    print(np.not_equal(list_tokenid_1, list_tokenid_2))