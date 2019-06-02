import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# from transformer.Layers import Conv
import hparams as hp


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

        self.duration_predictor = DurationPredictor()

    def LR(self, encoder_output, encoder_output_mask, duration_predictor_output, alpha):
        output = list()

        for i in range(encoder_output.size(0)):
            output.append(self.expand(
                encoder_output[i], duration_predictor_output[i], alpha))

        output = self.pad(output)

        return output

    def expand(self, one_batch, predicted, alpha):
        out = list()
        pad_length = list()

        for ele in predicted:
            pad_length.append(int(ele.data * alpha))

        for i, ele in enumerate(one_batch):
            [out.append(ele) for _ in range(pad_length[i] + 1)]

        out = torch.stack(out)

        return out

    def pad(self, input_ele):
        out_list = list()
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])
        # print(max_len)

        for batch in input_ele:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            # print(batch.size())
            # print(max_len-batch.size(0))
            # print(one_batch_padded.size())
            out_list.append(one_batch_padded)

        out_padded = torch.stack(out_list)

        return out_padded

    def forward(self, encoder_output, encoder_output_mask, alpha=1.0):
        duration_predictor_output_cal_loss = self.duration_predictor(encoder_output,
                                                                     encoder_output_mask)
        # print(duration_predictor_output_cal_loss)
        duration_predictor_output = torch.exp(
            duration_predictor_output_cal_loss)
        # print(duration_predictor_output)

        output = self.LR(encoder_output,
                         encoder_output_mask,
                         duration_predictor_output,
                         alpha)

        return output, duration_predictor_output_cal_loss


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self):
        super(DurationPredictor, self).__init__()

        self.input_size = hp.encoder_output_size
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

    def forward(self, encoder_output, encoder_output_mask):
        encoder_output = encoder_output * encoder_output_mask
        # encoder_output = encoder_output.contiguous().transpose(1, 2)

        out = self.conv_layer(encoder_output)
        # out = out.contiguous().transpose(1, 2)
        # print(out.size())

        out = self.linear_layer(out)
        out = out * encoder_output_mask[:, :, 0:1]
        # print(out)

        out = self.relu(out)
        # print(out)

        out = torch.log(out)
        # print("out size:", out.size())
        # print(encoder_output_mask[:, :, 0:1])
        # out = out * encoder_output_mask[:, :, 0:1]
        out = out.squeeze()
        # print("out size:", out.size())
        # print(out)

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
                 w_init='relu'):
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
        x = x .contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='relu'):
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


if __name__ == "__main__":
    # Test
    test_LR = LengthRegulator()
    # print(test_LR)

    test_encoder_output = torch.randn(2, 10, 384)
    test_encoder_output_mask_1 = torch.ones(2, 6, 384)
    test_encoder_output_mask_2 = torch.zeros(2, 4, 384)
    test_encoder_output_mask = torch.cat(
        (test_encoder_output_mask_1, test_encoder_output_mask_2), 1)

    test_output, _ = test_LR(test_encoder_output, test_encoder_output_mask)
    print(test_output.size())
