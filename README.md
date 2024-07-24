# Online Point Tracking in Ultrasound

This is the official code release for PIPsUS: Self-Supervised Dense Point Tracking in Ultrasound 
**[[Paper](https://arxiv.org/abs/2403.04969)]**

The paper is accepted by The 5th International Workshop of Advances in Simplifying Medical UltraSound (ASMUS) - a workshop held in conjunction with MICCAI 2024!

## Requirements

The lines below should set up a fresh environment with everything you need: 

```
conda create -n pips2 python=3.8
conda activate pips2
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install SimpleITK
```


## Citation


## Acknowledgement

We would like to thanks the authors of [PIPs++](https://arxiv.org/abs/2307.15055) and [RAFT](https://arxiv.org/pdf/2003.12039) for open-sourcing their codes and models. This work is built on top of their contribution.