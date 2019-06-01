from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import audio


def build_from_path(in_dir, out_dir, num_workers=16, tqdm=lambda x: x):
    # def build_from_path(in_dir, out_dir):
    '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

      Args:
        in_dir: The directory where you have downloaded the LJ Speech dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar

      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    '''

    # We use ProcessPoolExecutor to parallelize across processes. This is just an optimization and you
    # can omit it and just call _process_utterance on each input if you want.

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    # list_meta = []
    index = 1
    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            # print(parts[0])
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            futures.append(executor.submit(
                partial(_process_utterance, out_dir, index, wav_path, text)))
            # temp_tuple = _process_utterance(out_dir, index, wav_path, text)
            # print(temp_tuple[3])
            # list_meta.append(temp_tuple)
            if index % 100 == 0:
                print("Done %d" % index)
            index = index + 1
            # print(index)

    return [future.result() for future in tqdm(futures)]
    # return list_meta


def _process_utterance(out_dir, index, wav_path, text):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file

    Returns:
      A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
    '''

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    # print(len(spectrogram))
    # print(len(spectrogram[0]))
    # print(type(spectrogram))
    # print(np.shape(spectrogram))
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)
    # print(np.shape(mel_spectrogram.T))
    # print()

    # Write the spectrograms to disk:
    # spectrogram_filename = 'ljspeech-spec-%05d.npy' % index
    mel_filename = 'ljspeech-mel-%05d.npy' % index
    # np.save(os.path.join(out_dir, spectrogram_filename),
    #         spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    # return (spectrogram_filename, mel_filename, n_frames, text)
    return (mel_filename, n_frames, text)


# if __name__ == "__main__":
#     build_from_path("LJSpeech-1.1", "data")
