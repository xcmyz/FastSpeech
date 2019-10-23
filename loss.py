import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    """ FastSPeech Loss """

    def __init__(self):
        super(FastSpeechLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel, mel_postnet, duration_predicted, mel_target, duration_predictor_target):
        mel_target.requires_grad = False
        mel_loss = self.mse_loss(mel, mel_target)
        mel_postnet_loss = self.mse_loss(mel_postnet, mel_target)

        duration_predictor_target.requires_grad = False
        # duration_predictor_target = duration_predictor_target + 1
        # duration_predictor_target = torch.log(
        #     duration_predictor_target.float())

        # print(duration_predictor_target)
        # print(duration_predicted)

        duration_predictor_loss = self.l1_loss(
            duration_predicted, duration_predictor_target.float())

        return mel_loss, mel_postnet_loss, duration_predictor_loss
