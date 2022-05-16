import torch

from allennlp.common.util import lazy_groups_of
from allennlp.common.tqdm import Tqdm

def count(words, output_list):
    for d in words:
        for w in d["words"]:
            if w not in output_list:
                output_list.append(w)
    return output_list
    

def get_smaller_vocab(iterator, train_dataset):
    raw_train_generator = iterator(train_dataset, num_epochs=1, shuffle=True)
    train_generator = lazy_groups_of(raw_train_generator, 1)
    total = iterator.get_num_batches(train_dataset)
    output_list = {}
    for batch_group in Tqdm.tqdm(train_generator, total=total, ncols=75):
        batch = batch_group[0]
        output_list = count(batch["src_metadata"], output_list)
        if "labels" in batch:
            output_list = count(batch["tgt_metadata"], output_list)
    
    if 'start' not in output_list:
        output_list.append('start')
    if 'stop' not in output_list:
        output_list.append('stop')
    return output_list
        


def get_tgt_mask(tar: torch.Tensor):
    seq_len = tar.shape[1]
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=tar.device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def get_src_mask(tar: torch.Tensor):
    seq_len = tar.shape[1]
    mask = torch.zeros((seq_len, seq_len),device=tar.device).type(torch.bool)
    return mask


def remove_redudant(tokens: torch.Tensor):
    cur_tokens = torch.reshape(tokens, (-1,))
    tar_index = cur_tokens[0].data
    cur_mask = (cur_tokens != tar_index)
    cur_tokens = torch.masked_select(cur_tokens, cur_mask)
    cur_tokens = torch.reshape(cur_tokens, (tokens.shape[0], tokens.shape[1]-2))
    return cur_tokens

