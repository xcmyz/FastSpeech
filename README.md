# FastSpeech-Pytorch
The Implementation of FastSpeech Based on Pytorch.

## Model
<div style="text-align: center">
    <img src="img/fastspeech_structure.png" style="max-width:100%;">
</div>

## My Blog
- [FastSpeech Reading Notes](https://zhuanlan.zhihu.com/p/67325775)
- [Details and Rethinking of this Implementation](https://zhuanlan.zhihu.com/p/67939482)

## Prepare Dataset
1. Download and extract [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/).
2. Put LJSpeech dataset in `data`.
3. Unzip `alignments.zip`.
4. Put [Nvidia pretrained waveglow model](https://drive.google.com/file/d/1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx/view?usp=sharing) in the `waveglow/pretrained_model` and rename as `waveglow_256channels.pt`;
5. Run `python3 preprocess.py`.

## Training
Run `python3 train.py`.

## Evaluation
Run `python3 eval.py`.

## Notes
- In the paper of FastSpeech, authors use pre-trained Transformer-TTS model to provide the target of alignment. I didn't have a well-trained Transformer-TTS model so I use Tacotron2 instead.
- I use the same hyper-parameter as [FastSpeech2](https://arxiv.org/abs/2006.04558).
- The examples of audio are in `sample`.

## Reference
- [The Implementation of Tacotron Based on Tensorflow](https://github.com/keithito/tacotron)
- [The Implementation of Transformer Based on Pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- [The Implementation of Transformer-TTS Based on Pytorch](https://github.com/xcmyz/Transformer-TTS)
- [The Implementation of Tacotron2 Based on Pytorch](https://github.com/NVIDIA/tacotron2)