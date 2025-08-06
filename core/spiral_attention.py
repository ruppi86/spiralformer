import torch


def build_spiral_attention_mask(seq_len: int, device=None):
    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    for i in range(seq_len):
        mask[i, i] = True
        offset = 1
        while offset < seq_len:
            j_pos = i + offset
            j_neg = i - offset
            if j_pos < seq_len:
                mask[i, j_pos] = True
            if j_neg >= 0:
                mask[i, j_neg] = True
            offset *= 2
    return mask 