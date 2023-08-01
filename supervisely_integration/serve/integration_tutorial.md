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
class MyModel(sly.nn.inference.MaskTracking):
    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        # initialize model, load weights, load model on the device
        pass

    def predict(
        self,
        frames: List[np.ndarray],
        input_mask: mp.ndarray,
    ) -> List[np.ndarray]:
        # disable gradient calculation, pass input mask to your model, run it on given list of frames (frame-by-frame), save predictions to a list and update progress bar on each iteration
        # a simple code example
        torch.set_grad_enabled(False)
        results = []
        for frame in frames:
          prediction = self.model(input_mask, frame)
          results.append(prediction)
          self.video_interface._notify(task="mask tracking")
        return results
```

The superclass has a `serve()` method. For running the code on the Supervisely platform, `serve` method should be executed:

```python
model = MyModel()
m.serve()
```

The `serve` method deploys your model as a **REST API** service on the Supervisely platform. It means that other applications are able to send requests to your model and get predictions from it.