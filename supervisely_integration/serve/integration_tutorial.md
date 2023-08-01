---
description: >-
  Step-by-step tutorial on how to integrate custom video object segmentation
  neural network into Supervisely platform on the example of XMem.
---

# Custom video object segmentation model integration

## Introduction

In this tutorial you will learn how to integrate your video object segmentation model into Supervisely Ecosystem. Supervisely Python SDK allows to integrate models for numerous video object tracking tasks, such as tracking of bounding boxes, masks, keypoints, polylines, etc. This tutorial takes XMem video object segmentation model as an example and provides a complete instruction to integrate it as an application into Supervisely Ecosystem.

## Implementation details

To integrate your custom video object segmentation model, you need to subclass **`sly.nn.inference.MaskTracking`** and implement 2 methods:

* `load_on_device` method for loading the weights and initializing the model on a specific device. Takes a `model_dir` argument, which is a directory for all model files (like configs, weights, etc.), and a `device` argument - a torch.device like `cuda:0`, `cpu`.
* `predict` method for model inference. It takes a `frames` argument - a list of numpy arrays, which represents a set of video frames, and an `input_mask` agrument - a numpy array of shape (H, W), where 0 represents the background and other numbers represent target objects (for example, if you have 2 target objects, than input_mask array will consist of 0, 1 and 2 values).