import csv
import numpy as np




def prot_to_token(vocal, embeddings, protseqs, prot_len):
    # Load Ehsan Asgari's embeddings
    # Paper: http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0141287
    # Embedding: https://github.com/ehsanasgari/Deep-Proteomics

    def to_token(prot, n_vocabs=len(vocal)):
        # token_of_prot = np.array([embeddings[vocal['<unk>']]])
        # token_of_prot = np.repeat(token_of_prot, prot_len // 3, axis=0)

        protkn = []

        if len(prot) >= prot_len:
            prot3mers = [prot[ii:ii + 3] for ii in range(0, prot_len - prot_len % 3, 3)]
            if prot_len % 3 > 0:
                prot3mers.append('<unk>')

            for kmer_ in prot3mers:
                if kmer_ not in vocal:
                    print("kmer_ not in vocal", kmer_)
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
