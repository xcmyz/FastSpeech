import torch
import numpy as np
import shutil
import os

from utils import load_data, get_Tacotron2, get_WaveGlow
from utils import process_text, load_data
from data import ljspeech
import hparams as hp
import waveglow
import audio as Audio


def preprocess_ljspeech(filename):
    in_dir = filename
    out_dir = hp.mel_ground_truth
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    metadata = ljspeech.build_from_path(in_dir, out_dir)
    write_metadata(metadata, out_dir)

    shutil.move(os.path.join(hp.mel_ground_truth, "train.txt"),
                os.path.join("data", "train.txt"))


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write(m + '\n')


def main():
    path = os.path.join("data", "LJSpeech-1.1")
    preprocess_ljspeech(path)

    text_path = os.path.join("data", "train.txt")
    texts = process_text(text_path)

    if not os.path.exists(hp.alignment_path):
        os.mkdir(hp.alignment_path)
    else:
        return

    tacotron2 = get_Tacotron2()

    num = 0
    for ind, text in enumerate(texts[num:]):
        print(ind)

        character = text[0:len(text)-1]
        mel_gt_name = os.path.join(
            hp.mel_ground_truth, "ljspeech-mel-%05d.npy" % (ind+num+1))
        mel_gt_target = np.load(mel_gt_name)
        _, _, D = load_data(character, mel_gt_target, tacotron2)

        np.save(os.path.join(hp.alignment_path, str(
            ind+num) + ".npy"), D, allow_pickle=False)


if __name__ == "__main__":
    main()
