import torch
import torch.nn as nn

from transformer.Models import Encoder, Decoder
from transformer.Layers import Linear, PostNet
from Networks import LengthRegulator
import hparams as hp


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder()
        self.length_regulator = LengthRegulator()
        self.decoder = Decoder()

        self.mel_linear = Linear(hp.decoder_output_size, hp.num_mels)
        self.postnet = PostNet()

    def forward(self, src_seq, src_pos, length_target=None, alpha=1.0):
        encoder_output, encoder_mask = self.encoder(src_seq, src_pos)

        if self.training:
            length_regulator_output, decoder_pos, duration_predictor_output_cal_loss = self.length_regulator(
                encoder_output, encoder_mask, length_target, alpha)
            decoder_output = self.decoder(length_regulator_output, decoder_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output

            return mel_output, mel_output_postnet, duration_predictor_output_cal_loss
        else:
            length_regulator_output, decoder_pos = self.length_regulator(
                encoder_output, encoder_mask, alpha=alpha)

            decoder_output = self.decoder(length_regulator_output, decoder_pos)

            mel_output = self.mel_linear(decoder_output)
            mel_output_postnet = self.postnet(mel_output) + mel_output

            return mel_output, mel_output_postnet


if __name__ == "__main__":
    # Test
    test_encoder = Encoder()
    test_decoder = Decoder()
    # print(test_encoder)
    # print(test_decoder)

    test_src = torch.stack([torch.Tensor([1, 2, 4, 3, 2, 5, 0, 0]),
                            torch.Tensor([3, 4, 2, 6, 7, 1, 2, 3])]).long()
    test_pos = torch.stack([torch.Tensor([1, 2, 3, 4, 5, 6, 0, 0]),
                            torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8])]).long()
    test_target = torch.stack([torch.Tensor([0, 2, 3, 0, 3, 2, 1, 0]),
                               torch.Tensor([1, 2, 3, 2, 2, 0, 3, 6])])

    test_encoder_output, test_mask = test_encoder(test_src, test_pos)
    # print(test_mask)
    # print(test_encoder_output.size())

    test_decoder_output = test_decoder(test_encoder_output, test_pos)
    # print(test_decoder_output.size())

    test_fastspeech = FastSpeech()
    # print(test_fastspeech)

    test_mel_output, test_mel_output_postnet, test_duration_predictor_output_cal_loss = test_fastspeech(
        test_src, test_pos, test_target)
    print(test_mel_output.size())
    print(test_mel_output_postnet.size())
    print(test_duration_predictor_output_cal_loss.size())
    print(test_duration_predictor_output_cal_loss)
