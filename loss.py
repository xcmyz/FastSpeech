import torch
import torch.nn as nn


def cut(A, B, C):
    min_len = min(A.size(1), B.size(1), C.size(1))
    return A[:, 0:min_len, :], B[:, 0:min_len, :], C[:, 0:min_len, :]


class FastSpeechLoss(nn.Module):
    """ FastSPeech Loss """

    def __init__(self):
        super(FastSpeechLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, mel, mel_postnet, duration_predictor, mel_target, duration_predictor_target):
        mel_target.requires_grad = False
        mel, mel_postnet, mel_target = cut(mel, mel_postnet, mel_target)

        mel_loss = torch.abs(mel - mel_target)
        mel_loss = torch.mean(mel_loss)

        mel_postnet_loss = torch.abs(mel_postnet - mel_target)
        mel_postnet_loss = torch.mean(mel_postnet_loss)

        duration_predictor_target.requires_grad = False

        duration_predictor_loss = self.mse_loss(
            duration_predictor, duration_predictor_target)

        return mel_loss, mel_postnet_loss, duration_predictor_loss
