import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import math
import os

import hparams
import audio as Audio
from text import text_to_sequence
from utils import process_text, pad_1D, pad_2D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FastSpeechDataset(Dataset):
    """ LJSpeech """

    def __init__(self):
        self.text = process_text(os.path.join("data", "train.txt"))

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        mel_gt_name = os.path.join(
            hparams.mel_ground_truth, "ljspeech-mel-%05d.npy" % (idx+1))
        mel_gt_target = np.load(mel_gt_name)
        D = np.load(os.path.join(hparams.alignment_path, str(idx)+".npy"))

        character = self.text[idx][0:len(self.text[idx])-1]
        character = np.array(text_to_sequence(
            character, hparams.text_cleaners))

        sample = {"text": character,
                  "mel_target": mel_gt_target,
                  "D": D}

        return sample


def reprocess(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    Ds = [batch[ind]["D"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.shape[0])

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = np.array(src_pos)

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.shape[0])

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = np.array(mel_pos)

    texts = pad_1D(texts)
    Ds = pad_1D(Ds)
    mel_targets = pad_2D(mel_targets)

    out = {"text": texts,
           "mel_target": mel_targets,
           "D": Ds,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len}

    return out


def collate_fn(batch):
    len_arr = np.array([d["text"].shape[0] for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = int(math.sqrt(batchsize))

    cut_list = list()
    for i in range(real_batchsize):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(real_batchsize):
        output.append(reprocess(batch, cut_list[i]))

    return output


if __name__ == "__main__":
    # Test
    dataset = FastSpeechDataset()
    training_loader = DataLoader(dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 collate_fn=collate_fn,
                                 drop_last=True,
                                 num_workers=0)
    total_step = hparams.epochs * len(training_loader) * hparams.batch_size

    cnt = 0
    for i, batchs in enumerate(training_loader):
        for j, data_of_batch in enumerate(batchs):
            mel_target = torch.from_numpy(
                data_of_batch["mel_target"]).float().to(device)
            D = torch.from_numpy(data_of_batch["D"]).int().to(device)
            # print(mel_target.size())
            # print(D.sum())
            print(cnt)
            if mel_target.size(1) == D.sum().item():
                cnt += 1

    print(cnt)
