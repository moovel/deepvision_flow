# FlowNetPytorch
Pytorch implementation of FlowNet by Dosovitskiy et al.

This repository is a torch implementation of [FlowNet](http://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/), by [Alexey Dosovitskiy](http://lmb.informatik.uni-freiburg.de/people/dosovits/) et al. in PyTorch. This code is mainly inspired from [here](https://github.com/ClementPinard/FlowNetPytorch)

Deformable Convolutions as seen in this [paper](https://arxiv.org/abs/1703.06211), code inspired [by](https://github.com/ChunhuanLin/deform_conv_pytorch)

Following neural network models are currently provided :

 - **FlowNetS**
 - **FlowNetSBN**
 - **FlowNetC**
 - **FlowNetCBN**
 - **Def_FlowNetS**
 - **Def_FlowNetSBN**
 - **Def_FlowNetC**
 - **Def_FlowNetCBN**

## Prerequisite

```
pytorch >= 0.4.1
tensorboard-pytorch
tensorboardX >= 1.4
spatial-correlation-sampler>=0.0.8
imageio
argparse
```

## Training

* Example usage for FlowNetS :

```bash
python main.py /path/to/training_set/ -b8 -j8 -a flownets -s path/to/split/file
```
