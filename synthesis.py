import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pylab as plt
import os

import audio
from FastSpeech import FastSpeech
import hparams as hp
from text import text_to_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_data(data, figsize=(12, 4)):
    _, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto',
                       origin='bottom', interpolation='none')
    plt.savefig(os.path.join("img", "model_test.jpg"))


def get_waveglow():
    waveglow_path = os.path.join(hp.waveglow_path, 'waveglow_256channels.pt')
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()

    return waveglow


def synthesis_griffin_lim(text_seq, model, alpha=1.0, mode=""):
    text = text_to_sequence(text_seq, hp.text_cleaners)
    text = text + [0]
    text = np.stack([np.array(text)])
    text = torch.from_numpy(text).long().to(device)

    pos = torch.stack([torch.Tensor([i+1 for i in range(text.size(1))])])
    pos = pos.long().to(device)

    model.eval()
    with torch.no_grad():
        mel, mel_postnet = model(text, pos, alpha=alpha)

    mel = mel[0].cpu().numpy().T
    mel_postnet = mel_postnet[0].cpu().numpy().T
    plot_data([mel, mel_postnet])

    wav = audio.inv_mel_spectrogram(mel_postnet)
    print("Wav Have Been Synthesized.")

    if not os.path.exists("results"):
        os.mkdir("results")
    audio.save_wav(wav, os.path.join("results", text_seq + mode + ".wav"))


def synthesis_waveglow(text_seq, model, waveglow, alpha=1.0, mode=""):
    text = text_to_sequence(text_seq, hp.text_cleaners)
    text = text + [0]
    text = np.stack([np.array(text)])
    text = torch.from_numpy(text).long().to(device)

    pos = torch.stack([torch.Tensor([i+1 for i in range(text.size(1))])])
    pos = pos.long().to(device)

    model.eval()
    with torch.no_grad():
        _, mel_postnet = model(text, pos, alpha=alpha)
    with torch.no_grad():
        wav = waveglow.infer(mel_postnet, sigma=0.666)
    print("Wav Have Been Synthesized.")

    if not os.path.exists("results"):
        os.mkdir("results")
    audio.save_wav(wav[0].data.cpu().numpy(), os.path.join(
        "results", text_seq + mode + ".wav"))


if __name__ == "__main__":
    # Test
    model = nn.DataParallel(FastSpeech()).to(device)
    step_num = 148000
    checkpoint = torch.load(os.path.join(
        hp.checkpoint_path, 'checkpoint_%d.pth.tar' % step_num))
    model.load_state_dict(checkpoint['model'])
    print("Model Have Been Loaded.")

    words = "I am very happy to see you again."
    synthesis_griffin_lim(words, model, alpha=1.0, mode="normal")
    synthesis_griffin_lim(words, model, alpha=1.5, mode="slow")
    synthesis_griffin_lim(words, model, alpha=0.5, mode="quick")
    print("Synthesized.")

    # waveglow = get_waveglow()
    # synthesis_waveglow(words, model, waveglow,
    #                    alpha=1.0, mode="waveglow_normal")
    # print("Synthesized by Waveglow.")
