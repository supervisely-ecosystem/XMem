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

### Overall structure

The overall structure of the class we will implement looks like this:

```python
import supervisely as sly
import torch
import numpy as np

class MyModel(sly.nn.inference.MaskTracking):
    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        # initialize model, load weights, load model on device
        pass

    def predict(
        self,
        frames: List[np.ndarray],
        input_mask: np.ndarray,
    ) -> List[np.ndarray]:
        # a simple code example
        # disable gradient calculation
        torch.set_grad_enabled(False)
        results = []
        # pass input mask to your model, run it on given list of frames (frame-by-frame)
        for frame in frames:
          prediction = self.model(input_mask, frame)
          # save predictions to a list
          results.append(prediction)
          # update progress bar on each iteration
          self.video_interface._notify(task="mask tracking")
        return results
```

The superclass has a `serve` method. For running the code on the Supervisely platform, `serve` method should be executed:

```python
model = MyModel()
model.serve()
```

The `serve` method deploys your model as a **REST API** service on the Supervisely platform. It means that other applications are able to send requests to your model and get predictions from it.

## XMem video object segmentation model

Now let's implement the class specifically for XMem.

### Getting started

**Step 1.** Prepare `~/supervisely.env` file with credentials. [Learn more here](https://developer.supervisely.com/getting-started/basics-of-authentication#use-.env-file-recommended)

**Step 2.** Clone [repository](https://github.com/hkchengrex/XMem) with source code and create [Virtual Environment](https://docs.python.org/3/library/venv.html).

```bash
git clone https://github.com/hkchengrex/XMem.git
cd XMem
source .venv/bin/activate
pip3 install pip3 install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install supervisely==6.72.87
```

**Step 3.** Download model weights.
```bash
cd XMem
wget -P ./weights/ https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth
```

**Step 4.** Open the repository directory in Visual Studio Code.

```bash
code -r .
```

### Step-by-step implementation

**Defining imports and global variables**

```python
import supervisely as sly
import os
from dotenv import load_dotenv
from typing_extensions import Literal
from typing import List
import numpy as np
import torch
from model.network import XMem
from inference.inference_core import InferenceCore
from dataset.range_transform import im_normalization
from inference.interact.interactive_utils import index_numpy_to_one_hot_torch


# for debug, has no effect in production
load_dotenv("supervisely_integration/serve/debug.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

weights_location_path = "/weights/XMem.pth"
```

**1. load_on_device**

The following code creates XMem model with default hyperparameters recommended by original repository and defines resolution to which input video will be resized (we will use 480 as in original work). Also `load_on_device` will keep the model as a `self.model` and the device as `self.device` for further use:

```python
class XMemTracker(sly.nn.inference.MaskTracking):
    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        # define model configuration (default hyperparameters)
        self.config = {
            "top_k": 30,
            "mem_every": 5,
            "deep_update_every": -1,
            "enable_long_term": True,
            "enable_long_term_count_usage": True,
            "num_prototypes": 128,
            "min_mid_term_frames": 5,
            "max_mid_term_frames": 10,
            "max_long_term_elements": 10000,
        }
        # define resolution to which input video will be resized (was taken from original repository)
        self.resolution = 480
        # build model
        self.device = torch.device(device)
        self.model = XMem(self.config, weights_location_path, map_location=self.device).eval()
        self.model = self.model.to(self.device)
```

{% hint style="info" %}
For local debug we can load model weights from local storage, but in production we recommend to save weights to a Docker image.
{% endhint %}

**2. predict**

The core method for model inference. Here we are disabling gradient calculation, resizing input mask and frames via interpolation, inference XMem model frame-by-frame, saving postprocessed predictions to a list and updating progress bar on every iteration. Every prediction must be postprocessed to a multilabel numpy array of shape (H, W) - in other words it should have format similar to input mask. So if you are tracking 2 objects' masks on 20 frames, then frames will be a list of 20 numpy arrays, input_mask will be an array of shape (H, W) which consists of 0, 1 and 2 values, and the results variable will be a list of 20 numpy arrays, where each array will also be a numpy array of shape (H, W) which consists of 0, 1 and 2 values - similar to input mask. In the end of each iteration we update a progress bar via `self.video_interface._notify(task="mask tracking")` - it is necessary for app UI to look correctly:

```python
    def predict(
        self,
        frames: List[np.ndarray],
        input_mask: np.ndarray,
    ) -> List[np.ndarray]:
        # disable gradient calculation
        torch.set_grad_enabled(False)
        # empty cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # object IDs should be consecutive and start from 1 (0 represents the background)
        num_objects = len(np.unique(input_mask)) - 1
        # load processor
        processor = InferenceCore(self.model, config=self.config)
        processor.set_all_labels(range(1, num_objects + 1))
        # resize input mask
        original_width, original_height = input_mask.shape[1], input_mask.shape[0]
        scaler = min(original_width, original_height) / self.resolution
        resized_width = int(original_width / scaler)
        resized_height = int(original_height / scaler)
        input_mask = torch.from_numpy(input_mask)
        input_mask = input_mask.view(1, 1, input_mask.shape[0], input_mask.shape[1])
        input_mask = torch.nn.functional.interpolate(input_mask, (resized_height, resized_width), mode="nearest")
        input_mask = input_mask.squeeze().numpy()
        results = []
        # track input objects' masks
        with torch.cuda.amp.autocast(enabled=True):
            for i, frame in enumerate(frames):
                # preprocess frame
                frame = frame.transpose(2, 0, 1)
                frame = torch.from_numpy(frame)
                frame = torch.unsqueeze(frame, 0)
                frame = torch.nn.functional.interpolate(frame, (resized_height, resized_width), mode="nearest")
                frame = frame.squeeze()
                frame = frame.float().to(self.device) / 255
                frame = im_normalization(frame)
                # inference model on a specific frame
                if i == 0:
                    # preprocess input mask
                    input_mask = index_numpy_to_one_hot_torch(input_mask, num_objects + 1)
                    # the background mask is not fed into the model
                    input_mask = input_mask[1:]
                    input_mask = input_mask.to(self.device)
                    prediction = processor.step(frame, input_mask)
                else:
                    prediction = processor.step(frame)
                # postprocess prediction
                prediction = torch.argmax(prediction, dim=0)
                prediction = prediction.cpu().to(torch.uint8)
                prediction = prediction.view(1, 1, prediction.shape[0], prediction.shape[1])
                prediction = torch.nn.functional.interpolate(prediction, (original_height, original_width), mode="nearest")
                prediction = prediction.squeeze().numpy()
                # save predicted mask
                results.append(prediction)
                # update progress bar
                self.video_interface._notify(task="mask tracking")
        return results
```

{% hint style="info" %}
It is crucial to disable gradient calculation in predict method, not in load_on_device, because these methods are being executed in different threads, so if you try disabling gradient calculation in load_on_device method, then it will have no effect during inference, which can significantly increase GPU memory consumption.
{% endhint %}