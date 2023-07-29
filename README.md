<div align="center" markdown>

<img src="XXX"/>  

# XMem

state-of-the-art Video Object Segmentation (VOS) integrated into Supervisely Videos Labeling tool

<p align="center">
  <a href="#Original-work">Original work</a> •
  <a href="#Original-work">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a> •
    <a href="#Demo">Demo</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/XMem)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/XMem)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/XMem.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/XMem.png)](https://supervise.ly)

</div>

# Original work

## Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model

[Ho Kei Cheng](https://hkchengrex.github.io/), [Alexander Schwing](https://www.alexander-schwing.de/)

University of Illinois Urbana-Champaign

[[arXiv]](https://arxiv.org/abs/2207.07115) [[PDF]](https://arxiv.org/pdf/2207.07115.pdf) [[Project Page]](https://hkchengrex.github.io/XMem/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RXK5QsUo2-CnOiy5AOSjoZggPVHOPh1m?usp=sharing)

# Overview

🔥 We have successfully integrated the XMem Long-Term Video Object Segmentation Neural Network into our Supervisely platform.

Description from the original [paper](https://arxiv.org/pdf/2207.07115.pdf):

*"We present XMem, a video object segmentation architecture for long videos with unified feature memory stores inspired by the Atkinson-Shiffrin memory model. Prior work on video object segmentation typically only uses one type of feature memory. For videos longer than a minute, a single feature memory model tightly links memory consumption and accuracy. In contrast, following the Atkinson-Shiffrin model, we develop an architecture that incorporates multiple independent yet deeply-connected feature memory stores: a rapidly updated sensory memory, a high-resolution working memory, and a compact thus sustained long-term memory. Crucially, we develop a memory potentiation algorithm that routinely consolidates actively used working memory elements into the long-term memory, which avoids memory explosion and minimizes performance decay for long-term prediction. Combined with a new memory reading mechanism, XMem greatly exceeds state-of-the-art performance on long-video datasets while being on par with state-of-the-art methods (that do not work on long videos) on short-video datasets."*

https://user-images.githubusercontent.com/7107196/177921527-7a1bd593-2162-4598-9adf-f2112763fccf.mp4
>Handling long-term occlusion



# How To Run

# How To Use

# Demo

Very-long video; masked layer insertion:

https://user-images.githubusercontent.com/7107196/179089789-3d69adea-0405-4c83-ac28-45f59fe1e1c1.mp4

Source: https://www.youtube.com/watch?v=q5Xr0F4a0iU

Out-of-domain case:

https://user-images.githubusercontent.com/7107196/177920383-161f1da1-33f9-48b3-b8b2-09e450432e2b.mp4

Source: かぐや様は告らせたい ～天才たちの恋愛頭脳戦～ Ep.3; A-1 Pictures

### [[Failure Cases]](docs/FAILURE_CASES.md)

## Features

* Handle very long videos with limited GPU memory usage.
* Quite fast. Expect ~20 FPS even with long videos (hardware dependent).
* Come with a GUI (modified from [MiVOS](https://github.com/hkchengrex/MiVOS/tree/MiVOS-STCN)).

### Table of Contents

1. [Introduction](#introduction)
2. [Results](docs/RESULTS.md)
3. [Interactive GUI demo](docs/DEMO.md)
4. [Training/inference](#traininginference)
5. [Citation](#citation)

### Introduction

![framework](https://imgur.com/ToE2frx.jpg)

We frame Video Object Segmentation (VOS), first and foremost, as a *memory* problem.
Prior works mostly use a single type of feature memory. This can be in the form of network weights (i.e., online learning), last frame segmentation (e.g., MaskTrack), spatial hidden representation (e.g., Conv-RNN-based methods), spatial-attentional features (e.g., STM, STCN, AOT), or some sort of long-term compact features (e.g., AFB-URR).

Methods with a short memory span are not robust to changes, while those with a large memory bank are subject to a catastrophic increase in computation and GPU memory usage. Attempts at long-term attentional VOS like AFB-URR compress features eagerly as soon as they are generated, leading to a loss of feature resolution.

Our method is inspired by the Atkinson-Shiffrin human memory model, which has a *sensory memory*, a *working memory*, and a *long-term memory*. These memory stores have different temporal scales and complement each other in our memory reading mechanism. It performs well in both short-term and long-term video datasets, handling videos with more than 10,000 frames with ease.

### Training/inference

First, install the required python packages and datasets following [GETTING_STARTED.md](docs/GETTING_STARTED.md).

For training, see [TRAINING.md](docs/TRAINING.md).

For inference, see [INFERENCE.md](docs/INFERENCE.md).

### Citation

Please cite our paper if you find this repo useful!

```bibtex
@inproceedings{cheng2022xmem,
  title={{XMem}: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model},
  author={Cheng, Ho Kei and Alexander G. Schwing},
  booktitle={ECCV},
  year={2022}
}
```

Related projects that this paper is developed upon:
```bibtex
@inproceedings{cheng2021stcn,
  title={Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation},
  author={Cheng, Ho Kei and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={NeurIPS},
  year={2021}
}

@inproceedings{cheng2021mivos,
  title={Modular Interactive Video Object Segmentation: Interaction-to-Mask, Propagation and Difference-Aware Fusion},
  author={Cheng, Ho Kei and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={CVPR},
  year={2021}
}
```

We use f-BRS in the interactive demo: https://github.com/saic-vul/fbrs_interactive_segmentation

And if you want to cite the datasets:

<details> 
<summary>

bibtex

</summary>

```bibtex
@inproceedings{shi2015hierarchicalECSSD,
  title={Hierarchical image saliency detection on extended CSSD},
  author={Shi, Jianping and Yan, Qiong and Xu, Li and Jia, Jiaya},
  booktitle={TPAMI},
  year={2015},
}

@inproceedings{wang2017DUTS,
  title={Learning to Detect Salient Objects with Image-level Supervision},
  author={Wang, Lijun and Lu, Huchuan and Wang, Yifan and Feng, Mengyang 
  and Wang, Dong, and Yin, Baocai and Ruan, Xiang}, 
  booktitle={CVPR},
  year={2017}
}

@inproceedings{FSS1000,
  title = {FSS-1000: A 1000-Class Dataset for Few-Shot Segmentation},
  author = {Li, Xiang and Wei, Tianhan and Chen, Yau Pun and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{zeng2019towardsHRSOD,
  title = {Towards High-Resolution Salient Object Detection},
  author = {Zeng, Yi and Zhang, Pingping and Zhang, Jianming and Lin, Zhe and Lu, Huchuan},
  booktitle = {ICCV},
  year = {2019}
}

@inproceedings{cheng2020cascadepsp,
  title={{CascadePSP}: Toward Class-Agnostic and Very High-Resolution Segmentation via Global and Local Refinement},
  author={Cheng, Ho Kei and Chung, Jihoon and Tai, Yu-Wing and Tang, Chi-Keung},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{xu2018youtubeVOS,
  title={Youtube-vos: A large-scale video object segmentation benchmark},
  author={Xu, Ning and Yang, Linjie and Fan, Yuchen and Yue, Dingcheng and Liang, Yuchen and Yang, Jianchao and Huang, Thomas},
  booktitle = {ECCV},
  year={2018}
}

@inproceedings{perazzi2016benchmark,
  title={A benchmark dataset and evaluation methodology for video object segmentation},
  author={Perazzi, Federico and Pont-Tuset, Jordi and McWilliams, Brian and Van Gool, Luc and Gross, Markus and Sorkine-Hornung, Alexander},
  booktitle={CVPR},
  year={2016}
}

@inproceedings{denninger2019blenderproc,
  title={BlenderProc},
  author={Denninger, Maximilian and Sundermeyer, Martin and Winkelbauer, Dominik and Zidan, Youssef and Olefir, Dmitry and Elbadrawy, Mohamad and Lodhi, Ahsan and Katam, Harinandan},
  booktitle={arXiv:1911.01911},
  year={2019}
}

@inproceedings{shapenet2015,
  title       = {{ShapeNet: An Information-Rich 3D Model Repository}},
  author      = {Chang, Angel Xuan and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and Xiao, Jianxiong and Yi, Li and Yu, Fisher},
  booktitle   = {arXiv:1512.03012},
  year        = {2015}
}
```
</details>

Contact: <hkchengrex@gmail.com>
