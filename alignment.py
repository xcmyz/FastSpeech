# You need a reliable model for alignment.

# ------Test------ #

# For test, Multiply three there.

# import torch
# import numpy as np


# def get_alignment(text, pos):
#     test_out = torch.zeros(np.shape(pos)[0], np.shape(pos)[1])
#     for i_batch in range(np.shape(pos)[0]):
#         for i_ele in range(np.shape(pos)[1]):
#             if pos[i_batch][i_ele] != 0:
#                 test_out[i_batch][i_ele] = 3.0

#     return test_out


# ------Load Tacotron2------ #

import torch
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import os

import Tacotron2.hparams as hp_tacotron2
import Tacotron2.model as model_tacotron2
import Tacotron2.layers as layers_tacotron2
import Tacotron2.train as train_tacotron2
from text import text_to_sequence
import audio
import hparams as hp


def get_tacotron2():
    hparams = hp_tacotron2.create_hparams()
    hparams.sampling_rate = hp.sample_rate

    checkpoint_path = os.path.join("Tacotron2", os.path.join(
        "pre_trained_model", "tacotron2_statedict.pt"))

    tacotron2 = train_tacotron2.load_model(hparams)
    tacotron2.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    _ = tacotron2.cuda().eval().half()

    return tacotron2


def plot_data(data, figsize=(16, 4)):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       origin='bottom', interpolation='none')
    plt.savefig(os.path.join("img", "test_tacotron2.jpg"))


def get_tacotron2_alignment(text_seq, tacotron2):
    sequence = torch.autograd.Variable(text_seq)

    outputs = tacotron2.inference(sequence)
    alignment = outputs[3]
    alignment = alignment.float().data.cpu().numpy()[0]

    return alignment


def get_tacotron2_alignment_test(text_seq):
    hparams = hp_tacotron2.create_hparams()
    hparams.sampling_rate = hp.sample_rate

    checkpoint_path = os.path.join("Tacotron2", os.path.join(
        "pre_trained_model", "tacotron2_statedict.pt"))

    tacotron2 = train_tacotron2.load_model(hparams)
    tacotron2.load_state_dict(torch.load(checkpoint_path)["state_dict"])
    _ = tacotron2.cuda().eval().half()

    sequence = np.array(text_to_sequence(text_seq, hp.text_cleaners))[None, :]
    print("sequence size", np.shape(sequence))

    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    mel, mel_postnet, _, alignment = tacotron2.inference(sequence)

    plot_data((mel.float().data.cpu().numpy()[0],
               mel_postnet.float().data.cpu().numpy()[0],
               alignment.float().data.cpu().numpy()[0].T))

    wav = audio.inv_mel_spectrogram(mel_postnet.float().data.cpu().numpy()[0])
    audio.save_wav(wav, "test.wav")

    alignment = alignment.float().data.cpu().numpy()[0]
    print("alignment size", np.shape(alignment))

    get_D(alignment)

    return alignment


def get_D(alignment):
    D = np.array([0 for _ in range(np.shape(alignment)[1])])

    for i in range(np.shape(alignment)[0]):
        max_index = alignment[i].tolist().index(alignment[i].max())
        D[max_index] = D[max_index] + 1

    for i in range(np.shape(D)[0]):
        if D[i] > 0:
            D[i] = D[i] - 1

    return D


def get_one_alignment(text, tacotron2):
    alignment = get_tacotron2_alignment(text, tacotron2)
    D = get_D(alignment)
    D = torch.from_numpy(D)

    return D


def get_alignment(texts, tacotron2):
    out = torch.stack([get_one_alignment(texts[i:i+1], tacotron2)
                       for i in range(texts.size(0))])

    return out


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        inx = 0
        txt = []
        for line in f.readlines():
            cnt = 0
            for index, ele in enumerate(line):
                if ele == '|':
                    cnt = cnt + 1
                    if cnt == 2:
                        inx = index
                        end = len(line)
                        txt.append(line[inx+1:end-1])
                        break

        return txt


if __name__ == "__main__":
    # Test
    alignment = get_tacotron2_alignment_test(
        "I want to go to CMU to do research on deep learning.")
    print(alignment)
