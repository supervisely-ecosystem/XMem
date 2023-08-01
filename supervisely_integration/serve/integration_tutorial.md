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

**Creating necessary files and directories**

After cloning original repo we will create `supervisely_integration` folder, where all code for integration will be stored. There will be 2 directories - `docker` (we will put our Dockerfile here) and `serve` (app directory). Inside `serve` directory we will create `src` subdirectory and put `main.py` file there. Inside `serve` folder we will also create `debug.env` file - it will contain your [team id](https://developer.supervisely.com/getting-started/environment-variables#team_id):

```python
TEAM_ID=your_team_id
```

We will also create `requirements.txt` file, where all app dependancies will be stored:

```python
supervisely==6.72.87
--extra-index-url https://download.pytorch.org/whl/cu113
torch==1.12.1+cu113
torchvision==0.13.1+cu113
```

Now we can start coding our `main.py` file.

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

When `load_on_device` and `predict` methods are implemented, it is necessary to initialize our model class and execute `serve` method:

```python
model = XMemTracker()
model.serve()
```

## Debug in Supervisely platform

Once the code is written, it's time to test it right in the Supervisely platform as a debugging app.

First of all it is necessary to create `.vscode` folder and `launch.json` file inside this folder. Your `launch.json` file should contain the following:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Advanced Debug in Supervisely platform",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "main:model.app",
                "--app-dir",
                "./supervisely_integration/serve/src",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--ws",
                "websockets"
            ],
            "jinja": true,
            "justMyCode": false,
            "env": {
                "PYTHONPATH": "${workspaceFolder}/supervisely_integration/serve/src:${PYTHONPATH}",
                "LOG_LEVEL": "DEBUG",
                "ENV": "production",
                "DEBUG_WITH_SLY_NET": "1",
                "SLY_APP_DATA_DIR": "${workspaceFolder}/app_data"
            }
        }
    ]
}
```

You can read more about advanced debug mode [here](https://developer.supervisely.com/app-development/advanced/advanced-debugging).

After that: 

1. If you develop in a Docker container, you should run the container with `--cap_add=NET_ADMIN` option.

2. Install `sudo apt-get install wireguard iproute2` or `brew install wireguard-tools` for Mac.

3. Define your `TEAM_ID` in the `debug.env` file. *Actually there are other env variables that is needed, but they are already provided in `./vscode/launch.json` for you.

4. Switch the `launch.json` config to the `Advanced debug in Supervisely platform`:

![Advanced Debug in Supervisely](https://user-images.githubusercontent.com/31512713/224290229-5da93fd2-dc97-4911-abb5-66ce890485a2.png)

5. Run the code.

✅ It will deploy the model in the Supervisely platform as a REST API.

Here is how advanced debug mode launch looks like:

[https://github.com/supervisely-ecosystem/XMem/assets/91027877/6c07c37a-7b20-4e02-8fba-9f4fb5b98bef](https://user-images.githubusercontent.com/91027877/257574738-6c07c37a-7b20-4e02-8fba-9f4fb5b98bef.mp4)

After advanced debug launch you must be able to debug your app via `Develop & Debug` app:

[https://github.com/supervisely-ecosystem/XMem/assets/91027877/23198a4d-41cd-4ae7-a4e6-bba72c0da439](https://user-images.githubusercontent.com/91027877/257575433-23198a4d-41cd-4ae7-a4e6-bba72c0da439.mp4)

### Repository structure

The structure of [our GitHub repository](https://github.com/supervisely-ecosystem/XMem/tree/main) is the following:

```
├── LICENSE
├── README.md
├── app_data
│   └── models
├── dataset
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   └── range_transform.cpython-311.pyc
│   ├── range_transform.py
│   ├── reseed.py
│   ├── static_dataset.py
│   ├── tps.py
│   ├── util.py
│   └── vos_dataset.py
├── docs
│   ├── DEMO.md
│   ├── ECCV-logo.png
│   ├── FAILURE_CASES.md
│   ├── GETTING_STARTED.md
│   ├── INFERENCE.md
│   ├── PALETTE.md
│   ├── RESULTS.md
│   ├── TRAINING.md
│   ├── icon.png
│   ├── index.html
│   └── style.css
├── eval.py
├── inference
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   ├── inference_core.cpython-311.pyc
│   │   ├── kv_memory_store.cpython-311.pyc
│   │   └── memory_manager.cpython-311.pyc
│   ├── data
│   │   ├── __init__.py
│   │   ├── mask_mapper.py
│   │   ├── test_datasets.py
│   │   └── video_reader.py
│   ├── inference_core.py
│   ├── interact
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-311.pyc
│   │   │   └── interactive_utils.cpython-311.pyc
│   │   ├── fbrs
│   │   │   ├── LICENSE
│   │   │   ├── __init__.py
│   │   │   ├── controller.py
│   │   │   ├── inference
│   │   │   │   ├── __init__.py
│   │   │   │   ├── clicker.py
│   │   │   │   ├── evaluation.py
│   │   │   │   ├── predictors
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── base.py
│   │   │   │   │   ├── brs.py
│   │   │   │   │   ├── brs_functors.py
│   │   │   │   │   └── brs_losses.py
│   │   │   │   ├── transforms
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── base.py
│   │   │   │   │   ├── crops.py
│   │   │   │   │   ├── flip.py
│   │   │   │   │   ├── limit_longest_side.py
│   │   │   │   │   └── zoom_in.py
│   │   │   │   └── utils.py
│   │   │   ├── model
│   │   │   │   ├── __init__.py
│   │   │   │   ├── initializer.py
│   │   │   │   ├── is_deeplab_model.py
│   │   │   │   ├── is_hrnet_model.py
│   │   │   │   ├── losses.py
│   │   │   │   ├── metrics.py
│   │   │   │   ├── modeling
│   │   │   │   │   ├── __init__.py
│   │   │   │   │   ├── basic_blocks.py
│   │   │   │   │   ├── deeplab_v3.py
│   │   │   │   │   ├── hrnet_ocr.py
│   │   │   │   │   ├── ocr.py
│   │   │   │   │   ├── resnet.py
│   │   │   │   │   └── resnetv1b.py
│   │   │   │   ├── ops.py
│   │   │   │   └── syncbn
│   │   │   │       ├── LICENSE
│   │   │   │       ├── README.md
│   │   │   │       ├── __init__.py
│   │   │   │       └── modules
│   │   │   │           ├── __init__.py
│   │   │   │           ├── functional
│   │   │   │           │   ├── __init__.py
│   │   │   │           │   ├── _csrc.py
│   │   │   │           │   ├── csrc
│   │   │   │           │   │   ├── bn.h
│   │   │   │           │   │   ├── cuda
│   │   │   │           │   │   │   ├── bn_cuda.cu
│   │   │   │           │   │   │   ├── common.h
│   │   │   │           │   │   │   └── ext_lib.h
│   │   │   │           │   │   └── ext_lib.cpp
│   │   │   │           │   └── syncbn.py
│   │   │   │           └── nn
│   │   │   │               ├── __init__.py
│   │   │   │               └── syncbn.py
│   │   │   └── utils
│   │   │       ├── __init__.py
│   │   │       ├── cython
│   │   │       │   ├── __init__.py
│   │   │       │   ├── _get_dist_maps.pyx
│   │   │       │   ├── _get_dist_maps.pyxbld
│   │   │       │   └── dist_maps.py
│   │   │       ├── misc.py
│   │   │       └── vis.py
│   │   ├── fbrs_controller.py
│   │   ├── gui.py
│   │   ├── gui_utils.py
│   │   ├── interaction.py
│   │   ├── interactive_utils.py
│   │   ├── resource_manager.py
│   │   ├── s2m
│   │   │   ├── __init__.py
│   │   │   ├── _deeplab.py
│   │   │   ├── s2m_network.py
│   │   │   ├── s2m_resnet.py
│   │   │   └── utils.py
│   │   ├── s2m_controller.py
│   │   └── timer.py
│   ├── kv_memory_store.py
│   └── memory_manager.py
├── interactive_demo.py
├── merge_multi_scale.py
├── model
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-311.pyc
│   │   ├── aggregate.cpython-311.pyc
│   │   ├── cbam.cpython-311.pyc
│   │   ├── group_modules.cpython-311.pyc
│   │   ├── memory_util.cpython-311.pyc
│   │   ├── modules.cpython-311.pyc
│   │   ├── network.cpython-311.pyc
│   │   └── resnet.cpython-311.pyc
│   ├── aggregate.py
│   ├── cbam.py
│   ├── group_modules.py
│   ├── losses.py
│   ├── memory_util.py
│   ├── modules.py
│   ├── network.py
│   ├── resnet.py
│   └── trainer.py
├── requirements.txt
├── requirements_demo.txt
├── scripts
│   ├── __init__.py
│   ├── download_bl30k.py
│   ├── download_datasets.py
│   ├── download_models.sh
│   ├── download_models_demo.sh
│   ├── expand_long_vid.py
│   └── resize_youtube.py
├── supervisely_integration
│   ├── docker
│   │   ├── Dockerfile
│   │   └── publish.sh
│   └── serve
│       ├── README.md
│       ├── config.json
│       ├── debug.env
│       ├── requirements.txt
│       └── src
│           └── main.py
├── train.py
└── util
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-311.pyc
    │   ├── palette.cpython-311.pyc
    │   └── tensor_util.cpython-311.pyc
    ├── configuration.py
    ├── davis_subset.txt
    ├── image_saver.py
    ├── load_subset.py
    ├── log_integrator.py
    ├── logger.py
    ├── palette.py
    ├── tensor_util.py
    └── yv_subset.txt
```

Explanation:

* `supervisely_integration/serve/src/main.py` - main inference script
* `supervisely_integration/serve/README.md` - readme of your application, it is the main page of an application in Ecosystem with some images, videos, and how-to-use guides
* `supervisely_integration/serve/config.json` - configuration of the Supervisely application, which defines the name and description of the app, its context menu, icon, poster, and running settings
* `supervisely_integration/serve/requirements.txt` - all packages needed for debugging 
* `supervisely_integration/serve/debug.env` - file with variables used for debugging
* `supervisely_integration/docker` - directory with the custom Dockerfile for this application and the script that builds it and publishes it to the docker registry

### App configuration

App configuration is stored in `config.json` file. A detailed explanation of all possible fields is covered in this [Configuration Tutorial](https://developer.supervisely.com/app-development/basics/app-json-config/config.json). Let's check the config for our current app: 

```json
{
    "name": "XMem Video Object Segmentation",
    "type": "app",
    "version": "2.0.0",
    "categories": [
        "neural network",
        "videos",
        "segmentation & tracking",
        "serve"
    ],
    "description": "Semi-supervised, works with both long and short videos",
    "docker_image": "supervisely/xmem:1.0.1",
    "entrypoint": "python -m uvicorn main:model.app --app-dir ./supervisely_integration/serve/src --host 0.0.0.0 --port 8000 --ws websockets",
    "port": 8000,
    "task_location": "application_sessions",
    "icon": "https://github.com/supervisely-ecosystem/XMem/assets/119248312/bd2d09c8-db8c-4ae5-aec8-1f53f39afdcc",
    "icon_cover": true,
    "poster": "https://github.com/supervisely-ecosystem/XMem/assets/119248312/67188dc4-cc6b-47bd-b62e-d3d2b71ad7ac",
    "headless": true,
    "need_gpu": true,
    "gpu": "required",
    "instance_version": "6.7.40",
    "session_tags": [
        "sly_video_tracking"
    ],
    "community_agent": false,
    "allowed_shapes": [
        "bitmap",
        "polygon"
    ],
    "license": {
        "type": "GPL-3.0"
    }
}
```

Here is the explanation for the fields:

* `type` - type of the module in Supervisely Ecosystem
* `version` - version of Supervisely App Engine. Just keep it by default
* `name` - the name of the application
* `description` - the description of the application
* `categories` - these tags are used to place the application in the correct category in Ecosystem.
* `session_tags` - these tags will be assigned to every running session of the application. They can be used by other apps to find and filter all running sessions
* `need_gpu: true` - should be true if you want to use any `cuda` devices
*  `gpu: required` - app can be runned on both CPU and GPU devices, but it is recommended to use GPU for higher inference speed
* `community_agent: false` - this means that this app can not be run on the agents started by Supervisely team, so users have to connect their own computers and run the app only on their own agents. Only applicable in Community Edition. Enterprise customers use their private instances so they can ignore the current option
* `docker_image` - Docker container will be started from the defined Docker image, github repository will be downloaded and mounted inside the container.
* `entrypoint` - the command that starts our application in a container
* `port` - port inside the container
* `headless: true` means that the app has no User Interface
* `allowed_shapes` - shapes can be tracked with this model. In Supervisely masks can be represented by bitmap and polygon geometries.

