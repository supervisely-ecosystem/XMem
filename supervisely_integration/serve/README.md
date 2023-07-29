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
  <a href="#XMem-model-description">XMem model description</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/XMem)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/XMem)
[![views](https://app.supervise.ly/img/badges/views/supervisely-ecosystem/XMem.png)](https://supervise.ly)
[![runs](https://app.supervise.ly/img/badges/runs/supervisely-ecosystem/XMem.png)](https://supervise.ly)

</div>

# Overview

🔥 We have successfully integrated the XMem Long-Term Video Object Segmentation Neural Network into our Supervisely platform.

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

1. Go to Ecosystem and run the app [XMem Video Object Segmentation](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/XMem).  

<img data-key="sly-module-link" data-module-slug="supervisely-ecosystem/XMem Video Object Segmentation" src="https://github.com/supervisely-ecosystem/XMem/assets/119248312/e74e2bd9-f915-48b1-bb97-ee808326dff5" width="500px" style='padding-bottom: 20px'/> 

2. Run the app from **Neural Networks** page from category **segmentation & tracking videos**.

<img src="XXX"/>  

3. Run app on an agent with `GPU`.

<img src="XXX"/>

4. Use in `Videos Annotator`.

<img src="XXX"/>


# XMem model description

