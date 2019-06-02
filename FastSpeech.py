import torch
import torch.nn as nn

from transformer.Models import Encoder, Decoder
from transformer.Layers import Linear,PostNet
import hparams as hp


class Model(nn.Module):
    """ FastSpeech """

    def __init__(self):
        super(Model, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.mel_linear = Linear(hp.decoder_output_dim,hp.num_mels)
        self.postnet = PostNet()




if __name__ == "__main__":
    # Test
    test_encoder = Encoder()
    test_decoder = Decoder()
    print(test_encoder)
    print(test_decoder)
