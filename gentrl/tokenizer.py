import torch
import re

_aminos = ['I', 'L', 'V', 'F', 'M', 'C', 'A', 'G', 'P',
           'T', 'S', 'Y', 'W', 'Q', 'N', 'H', 'E', 'D', 'K', 'R']


def get_tokenizer_re(aminos):
    return re.compile('(' + '|'.join(aminos) + r'|\%\d\d|.)')


_aminos_re = get_tokenizer_re(_aminos)

__i2t = {
    0: 'unused', 1: 'I', 2: 'L', 3: 'V', 4: 'F', 5: 'M', 6: 'C', 7: 'A', 8: 'G', 9: 'P',
    10: 'T', 11: 'S', 12: 'Y', 13: 'W', 14: 'Q', 15: 'N', 16: 'H', 17: 'E', 18: 'D', 19: 'K', 20: 'R'
}

__t2i = {
    'unused': 0, 'I': 1, 'L': 2, 'V': 3, 'F': 4, 'M': 5, 'C': 6, 'A': 7, 'G': 8, 'P': 9,
    'T': 10, 'S': 11, 'Y': 12, 'W': 13, 'Q': 14, 'N': 15, 'H': 16, 'E': 17, 'D': 18, 'K': 19, 'R': 20
}


def aminos_tokenizer(line, aminos=None):
    """
    Tokenizes AMINOS
    Parameters:
         aminos: set of aminos for tokenization
    """
    if aminos is not None:
        reg = get_tokenizer_re(aminos)
    else:
        reg = _aminos_re
    return reg.split(line)[1::2]


def encode(sm_list, pad_size=50):
    """
    Encoder list of smiles to tensor of tokens
    """
    res = []
    lens = []
    for s in sm_list:
        tokens = ([1] + [__t2i[tok]
                         for tok in aminos_tokenizer(s)])[:pad_size - 1]
        lens.append(len(tokens))
        tokens += (pad_size - len(tokens)) * [2]
        res.append(tokens)

    return torch.tensor(res).long(), lens


def decode(tokens_tensor):
    """
    Decodes from tensor of tokens to list of smiles
    """

    aminos_res = []

    for i in range(tokens_tensor.shape[0]):
        cur_sm = ''
        for t in tokens_tensor[i].detach().cpu().numpy():
            if t == 2:
                break
            elif t > 2:
                cur_sm += __i2t[t]

        aminos_res.append(cur_sm)

    return aminos_res


def get_vocab_size():
    return len(__i2t)
