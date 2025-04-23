# Online Point Tracking in Ultrasound

This is the official code release for PIPsUS: Self-Supervised Point Tracking in Ultrasound 
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

## Model download link

You can download the PIPsUS model trained on a subset of the EchoNet here: [dropbox](https://www.dropbox.com/scl/fo/6e7upm8zlja53rvs1qt9f/AIoupYLL6STkYFj1Bws3GVU?rlkey=v7odkn1l0r30t08lrx6p27vev&st=8kcjke3d&dl=0). The folder contains a list of videos used in training, validation, and testing.

## Citation

Chen, Wanwen, Adam Schmidt, and Eitan Prisman. "PIPsUS: Self-supervised Point Tracking in Ultrasound." Simplifying Medical Ultrasound: 5th International Workshop, ASMUS 2024, Held in Conjunction with MICCAI 2024, Marrakesh, Morocco, October 6, 2024, Proceedings. Vol. 15186. Springer Nature, 2025.

## Acknowledgement

We would like to thank the authors of [PIPs++](https://arxiv.org/abs/2307.15055) and [RAFT](https://arxiv.org/pdf/2003.12039) for open-sourcing their codes and models. This work builds on their contribution.
