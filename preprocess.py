import os
from multiprocessing import cpu_count
from tqdm import tqdm
import hparams
from data import ljspeech


def preprocess_ljspeech(filename):
    in_dir = filename
    out_dir = "dataset"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    metadata = ljspeech.build_from_path(
        in_dir, out_dir, cpu_count(), tqdm=tqdm)
    write_metadata(metadata, out_dir)


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frames = sum([m[1] for m in metadata])
    hours = frames * hparams.frame_shift_ms / (3600 * 1000)
    print('Wrote %d utterances, %d frames (%.2f hours)' %
          (len(metadata), frames, hours))
    print('Max input length:  %d' % max(len(m[2]) for m in metadata))
    print('Max output length: %d' % max(m[1] for m in metadata))


def main():
    path = os.path.join("data", "LJSpeech-1.1")
    preprocess_ljspeech(path)


if __name__ == "__main__":
    main()
