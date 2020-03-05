import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np
import copy
import math

import hparams as hp
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor()

    def LR(self, x, duration_predictor_output, alpha=1.0, mel_max_length=None):
        output = list()

        for batch, expand_target in zip(x, duration_predictor_output):
            output.append(self.expand(batch, expand_target, alpha))

        if mel_max_length:
            output = utils.pad(output, mel_max_length)
        else:
            output = utils.pad(output)

        return output

    def expand(self, batch, predicted, alpha):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size*alpha), -1))
        out = torch.cat(out, 0)

        return out

    def rounding(self, num):
        if num - int(num) >= 0.5:
            return int(num) + 1
        else:
            return int(num)

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if self.training:
            output = self.LR(x, target, mel_max_length=mel_max_length)
            return output, duration_predictor_output
        else:
            for idx, ele in enumerate(duration_predictor_output[0]):
                duration_predictor_output[0][idx] = self.rounding(ele)
            output = self.LR(x, duration_predictor_output, alpha)
            mel_pos = torch.stack(
                [torch.Tensor([i+1 for i in range(output.size(1))])]).long().to(device)

            return output, mel_pos


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self):
        super(DurationPredictor, self).__init__()

        self.input_size = hp.d_model
        self.filter_size = hp.duration_predictor_filter_size
        self.kernel = hp.duration_predictor_kernel_size
        self.conv_output_size = hp.duration_predictor_filter_size
        self.dropout = hp.dropout

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("relu_1", nn.ReLU()),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("relu_2", nn.ReLU()),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)

        out = self.relu(out)

        out = out.squeeze()

        if not self.training:
            out = out.unsqueeze(0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class FFN(nn.Module):
    """
    Positionwise Feed-Forward Network
    """

    def __init__(self, num_hidden):
        """
        :param num_hidden: dimension of hidden 
        """
        super(FFN, self).__init__()
        self.w_1 = Conv(num_hidden, num_hidden * 4,
                        kernel_size=3, padding=1, w_init='relu')
        self.w_2 = Conv(num_hidden * 4, num_hidden, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, input_):
        # FFN Network
        x = input_
        x = self.w_2(torch.relu(self.w_1(x)))

        # residual connection
        x = x + input_

        # dropout
        x = self.dropout(x)

        # layer normalization
        x = self.layer_norm(x)

        return x


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden 
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query, mask=None, query_mask=None):
        # Get attention score
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        # Masking to ignore padding (key side)
        if mask is not None:
            attn = attn.masked_fill(mask, -2 ** 32 + 1)
            attn = torch.softmax(attn, dim=-1)
        else:
            attn = torch.softmax(attn, dim=-1)

        # Masking to ignore padding (query side)
        if query_mask is not None:
            attn = attn * query_mask

        # Dropout
        attn = self.attn_dropout(attn)

        # Get Context Vector
        result = torch.bmm(attn, value)

        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, num_hidden, h=2):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads 
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm_1 = nn.LayerNorm(num_hidden)

    def forward(self, memory, decoder_input, mask=None, query_mask=None):

        batch_size = memory.size(0)
        seq_k = memory.size(1)
        seq_q = decoder_input.size(1)

        # Repeat masks h times
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, seq_k)
            query_mask = query_mask.repeat(self.h, 1, 1)
        if mask is not None:
            mask = mask.repeat(self.h, 1, 1)

        # Make multihead
        key = self.key(memory).view(batch_size,
                                    seq_k,
                                    self.h,
                                    self.num_hidden_per_attn)
        value = self.value(memory).view(batch_size,
                                        seq_k,
                                        self.h,
                                        self.num_hidden_per_attn)
        query = self.query(decoder_input).view(batch_size,
                                               seq_q,
                                               self.h,
                                               self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1,
                                                        seq_k,
                                                        self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1,
                                                            seq_k,
                                                            self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1,
                                                            seq_q,
                                                            self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(
            key, value, query, mask=mask, query_mask=query_mask)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q,
                             self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(
            batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = torch.cat([decoder_input, result], dim=-1)

        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + decoder_input

        # Layer normalization
        result = self.layer_norm_1(result)

        return result, attns


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 n_head=hp.Head):
        super(FFTBlock, self).__init__()
        self.slf_attn = clones(Attention(d_model), hp.N)
        self.pos_ffn = clones(FFN(d_model), hp.N)

        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024,
                                                                                d_model,
                                                                                padding_idx=0), freeze=True)

    def forward(self, x, pos, return_attns=False):
        # Get character mask
        if self.training:
            c_mask = pos.ne(0).type(torch.float)
            mask = pos.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)
        else:
            c_mask, mask = None, None

        # Get positional embedding, apply alpha and add
        pos = self.pos_emb(pos)
        x = x + pos

        # Attention encoder-encoder
        attns = list()
        for slf_attn, ffn in zip(self.slf_attn, self.pos_ffn):
            x, attn = slf_attn(x, x, mask=mask, query_mask=c_mask)
            x = ffn(x)
            attns.append(attn)

        return x, attns
