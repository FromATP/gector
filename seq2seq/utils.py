import torch

from allennlp.common.util import lazy_groups_of
from allennlp.common.tqdm import Tqdm

def preload_vocab(iterator, train_dataset):
    raw_train_generator = iterator(train_dataset, num_epochs=1, shuffle=True)
    train_generator = lazy_groups_of(raw_train_generator, 1)
    total = iterator.get_num_batches(train_dataset)
    for batch_group in Tqdm.tqdm(train_generator, total=total, ncols=75):
        pass
    return True

def get_tgt_mask(tar: torch.Tensor):
    seq_len = tar.shape[1]
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=tar.device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def get_src_mask(tar: torch.Tensor):
    seq_len = tar.shape[1]
    mask = torch.zeros((seq_len, seq_len),device=tar.device).type(torch.bool)
    return mask
