<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/XMem/assets/119248312/67188dc4-cc6b-47bd-b62e-d3d2b71ad7ac"/>  

# XMem Video Object Segmentation

State-of-the-art Video Object Segmentation (VOS) integrated into Supervisely Videos Labeling tool

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#original-work">Original work</a> •
  <a href="#how-to-run">How To Run</a> •
  <a href="#xmem-model-framework">XMem model framework</a> •
  <a href="#citation">Citation</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/xmem/supervisely_integration/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/XMem)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/xmem/supervisely_integration/serve.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/xmem/supervisely_integration/serve.png)](https://supervise.ly)

</div>

# Overview

🔥 We have successfully integrated the XMem Long-Term Video Object Segmentation Neural Network into our Supervisely platform.

https://user-images.githubusercontent.com/12828725/257010007-c4df4dfc-4c38-4747-bdb6-6cd4ee7f031e.mp4

# Original work:

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


<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/xmem/supervisely_integration/serve" src="https://github.com/supervisely-ecosystem/XMem/assets/119248312/bdcd22bc-b67a-4597-8b1c-4f9baa913717" width="500px" style='padding-bottom: 20px'/> 

<br>

<img src="https://github.com/supervisely-ecosystem/XMem/assets/119248312/b1227abd-ac72-4ef0-b8d3-fb3e186839ed"/>


2. Or you can run it from the **Neural Networks** page from the category **Videos** -> **Segmentation & tracking**.

<img src="https://github.com/supervisely-ecosystem/XMem/assets/119248312/314789ab-657c-4279-837c-03bc58bc4fb7"/>

3. Run the app on an agent with `GPU`. For **Community Edition** - users have to run the app on their own GPU computer connected to the platform. Watch this [video tutorial](https://youtu.be/aO7Zc4kTrVg).

<img src="https://github.com/supervisely-ecosystem/XMem/assets/119248312/78a4a760-df76-4fcf-a6a0-049fa213e55f"/>

5. Segment the object with any interactive segmentation model (e.g. [Segment Anything](https://ecosystem.supervisely.com/apps/serve-segment-anything-model)) and press `Track` button in `Video Annotator`.

<img src="https://github.com/supervisely-ecosystem/XMem/assets/119248312/b6dc0f9e-1822-4862-9c65-83aaf629fb05"/>

Here is an example of creating input mask via Segment Anything Model and tracking this mask on multiple frames via XMem:

https://user-images.githubusercontent.com/91027877/257788666-9469cf60-6f40-42b1-8bc4-78d0a9896380.mp4

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

