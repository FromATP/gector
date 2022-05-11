import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def get_square_mask(tar: torch.Tensor):
    seq_len = tar.shape[1]
    mask = (torch.triu(torch.ones((seq_len, seq_len), device=tar.device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class AttentionalEncoder(nn.Module):

    def __init__(self, dict_size:int, output_dim:int, padding_idx:int):
        super(AttentionalEncoder, self).__init__()
        self.embedding_layer = nn.Embedding(dict_size, output_dim, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(output_dim, dropout=0.5)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, padding_mask: torch.Tensor=None):
        word_rep = self.embedding_layer(tokens.long())
        word_rep = self.pos_encoder(word_rep)
        outputs = self.transformer_encoder(word_rep, mask, src_key_padding_mask=padding_mask)
        return outputs


class SelfAttentionLayer(nn.Module):

    def __init__(self, dict_size:int, output_dim:int, padding_idx:int):
        super(SelfAttentionLayer, self).__init__()
        self.embedding_layer = nn.Embedding(dict_size, output_dim, padding_idx=padding_idx)
        self.attn = nn.MultiheadAttention(output_dim, num_heads=8, dropout=0.1, batch_first=True)
    
    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, padding_mask: torch.Tensor=None):
        word_rep = self.embedding_layer(tokens.long())
        outputs = self.attn(word_rep, word_rep, word_rep, 
                            attn_mask=mask, 
                            key_padding_mask=padding_mask, 
                            need_weights=False)
        return outputs


class AttentionalDecoder(nn.Module):

    def __init__(self, dict_size:int, output_dim:int, padding_idx:int):
        super(AttentionalDecoder, self).__init__()
        self.embedding_layer = nn.Embedding(dict_size, output_dim, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(output_dim, dropout=0.5)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=output_dim, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=4)
    
    def forward(self, tokens: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor, padding_mask: torch.Tensor=None):
        word_rep = self.embedding_layer(tokens.long())
        word_rep = self.pos_encoder(word_rep)
        outputs = self.transformer_decoder(word_rep, memory, mask, src_key_padding_mask=padding_mask)
        return outputs


class LinearLayer(nn.Module):

    def __init__(self, input_dim:int, output_dim:int):
        super(LinearLayer, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
    
    def forward(self, word_rep: torch.Tensor):
        outputs = self.layer(word_rep)
        return outputs