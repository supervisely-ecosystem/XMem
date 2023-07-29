<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/XMem/assets/119248312/67188dc4-cc6b-47bd-b62e-d3d2b71ad7ac"/>  

# XMem Video Object Segmentation

State-of-the-art Video Object Segmentation (VOS) integrated into Supervisely Videos Labeling tool

<p align="center">
  <a href="#Original-work">Original work</a> •
  <a href="#Original-work">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#How-To-Use">How To Use</a> •
  <a href="#Demo">Demo</a> •
  <a href="#XMem-model-framework">XMem model framework</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/XMem/supervisely_integration/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/XMem)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/xmem/supervisely_integration/serve.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/xmem/supervisely_integration/serve.png)](https://supervise.ly)

</div>

# Overview

🔥 We have successfully integrated the XMem Long-Term Video Object Segmentation Neural Network into our Supervisely platform.

https://user-images.githubusercontent.com/12828725/257010007-c4df4dfc-4c38-4747-bdb6-6cd4ee7f031e.mp4

## Original work:

Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model

[Ho Kei Cheng](https://hkchengrex.github.io/), [Alexander Schwing](https://www.alexander-schwing.de/)

University of Illinois Urbana-Champaign

[[arXiv]](https://arxiv.org/abs/2207.07115) [[PDF]](https://arxiv.org/pdf/2207.07115.pdf) [[Project Page]](https://hkchengrex.github.io/XMem/) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RXK5QsUo2-CnOiy5AOSjoZggPVHOPh1m?usp=sharing)
Original [paper](https://arxiv.org/pdf/2207.07115.pdf):

*"We present XMem, a video object segmentation architecture for long videos with unified feature memory stores inspired by the Atkinson-Shiffrin memory model. Prior work on video object segmentation typically only uses one type of feature memory. For videos longer than a minute, a single feature memory model tightly links memory consumption and accuracy. In contrast, following the Atkinson-Shiffrin model, we develop an architecture that incorporates multiple independent yet deeply-connected feature memory stores: a rapidly updated sensory memory, a high-resolution working memory, and a compact thus sustained long-term memory. Crucially, we develop a memory potentiation algorithm that routinely consolidates actively used working memory elements into the long-term memory, which avoids memory explosion and minimizes performance decay for long-term prediction. Combined with a new memory reading mechanism, XMem greatly exceeds state-of-the-art performance on long-video datasets while being on par with state-of-the-art methods (that do not work on long videos) on short-video datasets."*

https://user-images.githubusercontent.com/7107196/177921527-7a1bd593-2162-4598-9adf-f2112763fccf.mp4
>Handling long-term occlusion

# How To Run

0. This video object segmentation app is started by default in most cases by an instance administrator. If it isn't available in the video labeling tool, you can contact your Supervisely instance admin or run this app by yourself by following the steps below.

1. Go to Ecosystem page and find the app [XMem Video Object Segmentation](https://ecosystem.supervisely.com/apps/xmem/supervisely_integration/serve).  

![](https://github.com/supervisely-ecosystem/XMem/assets/12828725/68cd8c59-c2ff-47d6-bf71-072ea33ed9a3)

<br>

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/xmem/supervisely_integration/serve" src="https://github.com/supervisely-ecosystem/XMem/assets/12828725/0e62d1dc-4d1f-4014-a188-29be4eff9d14" width="300px" style='padding-bottom: 20px'/> 

2. Or you can run it from the **Neural Networks** page from the category **Videos** -> **Segmentation & tracking**.

<img src="https://github.com/supervisely-ecosystem/XMem/assets/12828725/525e2fbb-e9ee-4393-8c18-324498a0fa4a" width="600"/>

<br>

<img src="https://github.com/supervisely-ecosystem/XMem/assets/12828725/7d71e734-7e31-42b5-969e-34b3adf07204" width="600"/>  

3. Run the app on an agent with `GPU`. For **Community Edition** - users have to run the app on their own GPU computer connected to the platform. Watch this [video tutorial](https://youtu.be/aO7Zc4kTrVg).

4. Segment the object with any interactive segmentation model (e.g. [Segment Anything](https://ecosystem.supervisely.com/apps/serve-segment-anything-model)) and press `Track` button in `Video Annotator`.


# XMem model framework

![framework](https://imgur.com/ToE2frx.jpg)

We frame Video Object Segmentation (VOS), first and foremost, as a *memory* problem.
Prior works mostly use a single type of feature memory. This can be in the form of network weights (i.e., online learning), last frame segmentation (e.g., MaskTrack), spatial hidden representation (e.g., Conv-RNN-based methods), spatial-attentional features (e.g., STM, STCN, AOT), or some sort of long-term compact features (e.g., AFB-URR).

Methods with a short memory span are not robust to changes, while those with a large memory bank are subject to a catastrophic increase in computation and GPU memory usage. Attempts at long-term attentional VOS like AFB-URR compress features eagerly as soon as they are generated, leading to a loss of feature resolution.

Our method is inspired by the Atkinson-Shiffrin human memory model, which has a *sensory memory*, a *working memory*, and a *long-term memory*. These memory stores have different temporal scales and complement each other in our memory reading mechanism. It performs well in both short-term and long-term video datasets, handling videos with more than 10,000 frames with ease.

# Citation 

Please cite our paper if you find this repo useful!

```bibtex
@inproceedings{cheng2022xmem,
  title={{XMem}: Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model},
  author={Cheng, Ho Kei and Alexander G. Schwing},
  booktitle={ECCV},
  year={2022}
}
```

