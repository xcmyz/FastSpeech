import os
import audio


def test():
    wavs_path = os.path.join("data", "LJSpeech-1.1")
    wavs_path = os.path.join(wavs_path, "wavs")
    wav_path = os.path.join(wavs_path, "LJ001-0001.wav")
    wav = audio.load_wav(wav_path)
    mel_spec = audio.melspectrogram(wav)
    wav_after_inv = audio.inv_mel_spectrogram(mel_spec)
    audio.save_wav(wav_after_inv, "test.wav")


if __name__ == "__main__":
    # Test
    test()
