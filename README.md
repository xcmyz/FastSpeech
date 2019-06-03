# FastSpeech-Pytorch
The Implementation of FastSpeech Based on Pytorch.

## Model
<div align="center">
<img src="img/model.png">
</div>

## My Blog
- [FastSpeech Reading Notes(Chinese)](https://zhuanlan.zhihu.com/p/67325775)
- [Details of this Implementation](https://zhuanlan.zhihu.com/p/67325775)

## Train
1. Download and extract [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).
2. Put LJSpeech dataset in `data`.
3. Run `preprocess.py`.
4. If you want to get the target of alignment before training(It will speed up the training process greatly), you need download the pre-trained Tacotron2 model published by NVIDIA [here](https://drive.google.com/uc?export=download&confirm=XAHL&id=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA).
5. Put the pre-trained Tacotron2 model in `Tacotron2/pre_trained_model`
6. Run `alignment.py`, it will take long time.
7. Change `pre_target = True` in `hparam.py`.
8. Run `train.py`.
9. The tacotron2 outputs of mel spectrogram and alignment are shown as follow:
<div align="center">
<img src="img/test_tacotron2.jpg">
</div>

## Dependencies
- python 3.6
- pytorch 1.1.0
- numpy 1.16.2
- scipy 1.2.1
- librosa 0.6.3
- inflect 2.1.0
- matplotlib 2.2.2

## Notes
- If you don't prepare the target of alignment before training, the process of training would be very long.
- In the paper of Transformer-TTS, authors use pre-trained Transformer-TTS to provide the target of alignment. I didn't have a well-trained Transformer-TTS model so I use Tacotron2 instead.
- If you want to use another model to get targets of alignment, you need rewrite `alignment.py`.
- The returned value of `alignment.py` is a tensor whose value is the multiple that encoder's outputs are supposed to be expanded by.
- For example: 
```python
test_target = torch.stack([torch.Tensor([0, 2, 3, 0, 3, 2, 1, 0, 0, 0]),
                           torch.Tensor([1, 2, 3, 2, 2, 0, 3, 6, 3, 5])])
```

## Reference
- [The Implementation of Tacotron Based on Tensorflow](https://github.com/keithito/tacotron)
- [The Implementation of Transformer Based on Pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- [The Implementation of Transformer-TTS Based on Pytorch](https://github.com/xcmyz/Transformer-TTS)
- [The Implementation of Tacotron2 Based on Pytorch](https://github.com/NVIDIA/tacotron2)
