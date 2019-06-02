import torch
import torch.nn as nn
import numpy as np

import transformer.Constants as Constants
# from transformer.Layers import EncoderLayer, DecoderLayer, EncoderPreNet, PreNet, PostNet, Linear
from transformer.Layers import FFTBlock, PreNet, PostNet, Linear
from text.symbols import symbols
import hparams as hp


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


# def get_subsequent_mask(seq):
#     # subsequent：随后的
#     ''' For masking out the subsequent info. '''

#     sz_b, len_s = seq.size()
#     subsequent_mask = torch.triu(
#         torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
#     subsequent_mask = subsequent_mask.unsqueeze(
#         0).expand(sz_b, -1, -1)  # b x ls x ls

#     return subsequent_mask


class Encoder(nn.Module):
    ''' Encoder '''

    def __init__(self,
                 n_src_vocab=len(symbols)+1,
                 len_max_seq=hp.max_sep_len,
                 d_word_vec=hp.word_vec_dim,
                 n_layers=hp.encoder_n_layer,
                 n_head=hp.encoder_head,
                 d_k=64,
                 d_v=64,
                 d_model=hp.word_vec_dim,
                 d_inner=hp.encoder_conv1d_filter_size,
                 dropout=hp.dropout):

        super(Encoder, self).__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        # self.encoder_prenet = EncoderPreNet()

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # print("slf_attn_mask:\n", slf_attn_mask)
        # print("non_pad_mask:\n", non_pad_mask)

        # -- Forward
        # print(src_pos)
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        # enc_output = self.encoder_prenet(src_seq) + self.position_enc(src_pos)
        # enc_output = self.src_word_emb(src_seq)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        # if return_attns:
        #     return enc_output, enc_slf_attn_list

        return enc_output, non_pad_mask


# class Decoder(nn.Module):
#     ''' A decoder model with self attention mechanism. '''

#     def __init__(self,
#                  num_mel=80,
#                  n_tgt_vocab=1024,
#                  len_max_seq=2048,
#                  d_word_vec=512,
#                  n_layers=6,
#                  n_head=8,
#                  d_k=64,
#                  d_v=64,
#                  d_model=512,
#                  d_inner=2048,
#                  dropout=0.1):

#         super().__init__()
#         n_position = len_max_seq + 1

#         # self.tgt_word_emb = nn.Embedding(
#         #     n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)

#         self.decoder_prenet = PreNet(num_mel, d_word_vec*2, d_word_vec)

#         self.position_enc = nn.Embedding.from_pretrained(
#             get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
#             freeze=True)

#         self.layer_stack = nn.ModuleList([
#             DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
#             for _ in range(n_layers)])

#     def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, mel_tgt, return_attns=False):

#         dec_slf_attn_list, dec_enc_attn_list = [], []

#         # -- Prepare masks
#         non_pad_mask = get_non_pad_mask(tgt_seq)

#         slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)

#         slf_attn_mask_keypad = get_attn_key_pad_mask(
#             seq_k=tgt_seq, seq_q=tgt_seq)

#         slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

#         dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

#         # -- Forward
#         # dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

#         dec_output = self.decoder_prenet(mel_tgt) + self.position_enc(tgt_pos)

#         for dec_layer in self.layer_stack:
#             dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
#                 dec_output, enc_output,
#                 non_pad_mask=non_pad_mask,
#                 slf_attn_mask=slf_attn_mask,
#                 dec_enc_attn_mask=dec_enc_attn_mask)

#             if return_attns:
#                 dec_slf_attn_list += [dec_slf_attn]
#                 dec_enc_attn_list += [dec_enc_attn]

#         if return_attns:
#             return dec_output, dec_slf_attn_list, dec_enc_attn_list
#         return dec_output,


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self,
                 len_max_seq=hp.max_sep_len,
                 d_word_vec=hp.word_vec_dim,
                 n_layers=hp.decoder_n_layer,
                 n_head=hp.decoder_head,
                 d_k=64,
                 d_v=64,
                 d_model=hp.word_vec_dim,
                 d_inner=hp.decoder_conv1d_filter_size,
                 dropout=hp.dropout):

        super(Decoder, self).__init__()

        n_position = len_max_seq + 1

        # self.src_word_emb = nn.Embedding(
        #     n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        # self.encoder_prenet = EncoderPreNet()

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        # if return_attns:
        #     return dec_output, dec_slf_attn_list

        return dec_output


# class Transformer(nn.Module):
#     ''' A sequence to sequence model with attention mechanism. '''

#     def __init__(
#             self,
#             n_src_vocab, n_tgt_vocab, len_max_seq,
#             d_word_vec=512, d_model=512, d_inner=2048,
#             n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
#             tgt_emb_prj_weight_sharing=True,
#             emb_src_tgt_weight_sharing=True):

#         super().__init__()

#         self.encoder = Encoder(
#             n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
#             d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
#             n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
#             dropout=dropout)

#         self.decoder = Decoder(
#             n_tgt_vocab=n_tgt_vocab, len_max_seq=len_max_seq,
#             d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
#             n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
#             dropout=dropout)

#         self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
#         nn.init.xavier_normal_(self.tgt_word_prj.weight)

#         assert d_model == d_word_vec, \
#             'To facilitate the residual connections, \
#          the dimensions of all module outputs shall be the same.'

#         if tgt_emb_prj_weight_sharing:
#             # Share the weight matrix between target word embedding & the final logit dense layer
#             self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
#             self.x_logit_scale = (d_model ** -0.5)
#         else:
#             self.x_logit_scale = 1.

#         if emb_src_tgt_weight_sharing:
#             # Share the weight matrix between source & target word embeddings
#             assert n_src_vocab == n_tgt_vocab, \
#                 "To share word embedding table, the vocabulary size of src/tgt shall be the same."
#             self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

#     def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

#         tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

#         enc_output, *_ = self.encoder(src_seq, src_pos)
#         dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
#         seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

#         return seq_logit.view(-1, seq_logit.size(2))


# class TransformerTTS(nn.Module):
#     """ TTS model based on Transformer """

#     def __init__(self, num_mel=80, embedding_size=512):
#         super(TransformerTTS, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()
#         self.postnet = PostNet()
#         self.stop_linear = Linear(embedding_size, 1, w_init='sigmoid')
#         self.mel_linear = Linear(embedding_size, num_mel)

#     def forward(self, src_seq, src_pos, tgt_seq, tgt_pos, mel_tgt, return_attns=False):
#         encoder_output = self.encoder(src_seq, src_pos)
#         decoder_output = self.decoder(
#             tgt_seq, tgt_pos, src_seq, encoder_output[0], mel_tgt)
#         decoder_output = decoder_output[0]

#         mel_output = self.mel_linear(decoder_output)
#         mel_output_postnet = self.postnet(mel_output) + mel_output

#         stop_token = self.stop_linear(decoder_output)
#         stop_token = stop_token.squeeze(2)

#         return mel_output, mel_output_postnet, stop_token
