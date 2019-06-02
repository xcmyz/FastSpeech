# FastSpeech-Pytorch
The Implementation of FastSpeech Based on Pytorch.

## Model
<div align="center">
<img src="img/model.png">
</div>

## My Blog
[FastSpeech Reading Notes](https://zhuanlan.zhihu.com/p/67325775)

## Train
1. Put [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/) in `data`.
2. Run `preprocess.py`
3. Run `train.py`

## Dependencies
- python 3.6
- pytorch 1.1.0
- numpy 1.16.2
- scipy 1.2.1
- librosa 0.6.3
- inflect 2.1.0
- matplotlib 2.2.2

## Notes
- You need rewrite `alignment.py` because I didn't provide a pre-trained model to get attention matrix. I suggest you can use [the trained model which published by NVIDIA](https://drive.google.com/file/d/1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA/view).
- The returned value of `alignment.py` is a tensor whose value is the multiple that encoder's outputs are supposed to be expanded by. For example:
"`python3
test_target = torch.stack([torch.Tensor([0, 2, 3, 0, 3, 2, 1, 0, 0, 0]),
                           torch.Tensor([1, 2, 3, 2, 2, 0, 3, 6, 3, 5])])
"`

## Reference
- [The Implementation of Tacotron Based on Tensorflow](https://github.com/keithito/tacotron)
- [The Implementation of Transformer Based on Pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- [The Implementation of Transformer-TTS Based on Pytorch](https://github.com/xcmyz/Transformer-TTS)
