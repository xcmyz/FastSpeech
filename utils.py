import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

import Tacotron2
import text
import hparams


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def plot_data(data, figsize=(12, 4)):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       origin='bottom', interpolation='none')

    if not os.path.exists("img"):
        os.mkdir("img")
    plt.savefig(os.path.join("img", "model_test.jpg"))


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()

    return mask


def get_WaveGlow():
    waveglow_path = os.path.join("waveglow", "pretrained_model")
    waveglow_path = os.path.join(waveglow_path, "waveglow_256channels.pt")
    wave_glow = torch.load(waveglow_path)['model']
    wave_glow = wave_glow.remove_weightnorm(wave_glow)
    wave_glow.cuda().eval()
    for m in wave_glow.modules():
        if 'Conv' in str(type(m)):
            setattr(m, 'padding_mode', 'zeros')

    return wave_glow


def get_Tacotron2():
    checkpoint_path = "tacotron2_statedict.pt"
    checkpoint_path = os.path.join(os.path.join(
        "Tacotron2", "pretrained_model"), checkpoint_path)

    model = Tacotron2.model.Tacotron2(
        Tacotron2.hparams.create_hparams()).cuda()
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()

    return model


def get_D(alignment):
    D = np.array([0 for _ in range(np.shape(alignment)[1])])

    for i in range(np.shape(alignment)[0]):
        max_index = alignment[i].tolist().index(alignment[i].max())
        D[max_index] = D[max_index] + 1

    return D


def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        out_list = list()
        max_len = mel_max_length
        for i, batch in enumerate(input_ele):
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded
    else:
        out_list = list()
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

        for i, batch in enumerate(input_ele):
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded


def load_data(txt, mel, model):
    character = text.text_to_sequence(txt, hparams.text_cleaners)
    character = torch.from_numpy(np.stack([np.array(character)])).long().cuda()

    text_length = torch.Tensor([character.size(1)]).long().cuda()
    mel = torch.from_numpy(np.stack([mel.T])).float().cuda()
    max_len = mel.size(2)
    output_length = torch.Tensor([max_len]).long().cuda()

    inputs = character, text_length, mel, max_len, output_length

    with torch.no_grad():
        [_, mel_tacotron2, _, alignment], cemb = model.forward(inputs)

    alignment = alignment[0].cpu().numpy()
    cemb = cemb[0].cpu().numpy()

    D = get_D(alignment)
    D = np.array(D)

    mel_tacotron2 = mel_tacotron2[0].cpu().numpy()

    return mel_tacotron2, cemb, D


def load_data_from_tacotron2(txt, model):
    character = text.text_to_sequence(txt, hparams.text_cleaners)
    character = torch.from_numpy(np.stack([np.array(character)])).long().cuda()

    with torch.no_grad():
        [_, mel, _, alignment], cemb = model.inference(character)

    alignment = alignment[0].cpu().numpy()
    cemb = cemb[0].cpu().numpy()

    D = get_D(alignment)
    D = np.array(D)

    mel = mel[0].cpu().numpy()

    return mel, cemb, D
