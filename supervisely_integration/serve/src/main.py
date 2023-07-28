import supervisely as sly
import os
from dotenv import load_dotenv
from typing_extensions import Literal
from typing import Any, Dict, List
from supervisely.nn.inference import MaskTracking
import numpy as np
import torch
import torch.nn.functional as F
from model.network import XMem
from inference.inference_core import InferenceCore
from dataset.range_transform import im_normalization
from inference.interact.interactive_utils import torch_prob_to_numpy_mask, index_numpy_to_one_hot_torch


load_dotenv("supervisely_integration/serve/debug.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

weights_location_path = "/weights/XMem.pth"
# for debug
# weights_location_path = "./weights/XMem.pth"

class XMemTracker(MaskTracking):
    def load_on_device(
        self,
        model_dir: str,
        device: Literal["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"] = "cpu",
    ):
        self.device = torch.device(device)
        # disable gradient calculation
        torch.set_grad_enabled(False)
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
            "max_long_term_elements": 15000,
        }
        # build model
        self.model = XMem(self.config, weights_location_path, map_location=self.device).eval()
        self.model = self.model.to(self.device)
        # model quantization
        self.model.half()

    def predict(
            self,
            frames: List[np.ndarray],
            input_mask: np.ndarray,
    ):
        # object IDs should be consecutive and start from 1 (0 represents the background)
        num_objects = len(np.unique(input_mask))
        # load processor
        processor = InferenceCore(self.model, config=self.config)
        processor.set_all_labels(range(1, num_objects))
        # resize frames and input mask
        original_width, original_height = frames[0].shape[1], frames[0].shape[0]
        scaler = min(original_width, original_height) / 480
        resized_width = int(original_width / scaler)
        resized_height = int(original_height / scaler)
        # input_mask = np.resize(input_mask, (resized_height, resized_width))
        input_mask = torch.from_numpy(input_mask)
        input_mask = torch.unsqueeze(input_mask, 0)
        input_mask = torch.unsqueeze(input_mask, 0)
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
                frame = frame.squeeze().numpy()
                frame = torch.from_numpy(frame).float().to(self.device) / 255
                frame = im_normalization(frame)
                # inference model on specific frame
                if i == 0:
                    # preprocess input mask
                    input_mask = index_numpy_to_one_hot_torch(input_mask, num_objects)
                    input_mask = input_mask[1:]
                    input_mask = input_mask.to(self.device)
                    prediction = processor.step(frame, input_mask)
                else:
                    prediction = processor.step(frame)
                # postprocess prediction
                prediction = torch_prob_to_numpy_mask(prediction)
                prediction = torch.from_numpy(prediction)
                prediction = torch.unsqueeze(prediction, 0)
                prediction = torch.unsqueeze(prediction, 0)
                prediction = torch.nn.functional.interpolate(prediction, (original_height, original_width), mode="nearest")
                prediction = prediction.squeeze().numpy()
                # save predicted mask
                results.append(prediction)
                # update progress bar
                self.video_interface._notify(task="mask tracking")
        return results


model = XMemTracker(model_dir="./app_data/")
model.serve()