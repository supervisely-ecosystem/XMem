import supervisely as sly
import os
from dotenv import load_dotenv
from typing_extensions import Literal
from typing import Generator, List
import numpy as np
import torch
from model.network import XMem
from inference.inference_core import InferenceCore
from dataset.range_transform import im_normalization
from inference.interact.interactive_utils import index_numpy_to_one_hot_torch


# for debug, has no effect in production
load_dotenv("supervisely_integration/serve/debug.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

os.environ["SMART_CACHE_TTL"] = str(5 * 60)
os.environ["SMART_CACHE_SIZE"] = str(512)

weights_location_path = "/weights/XMem.pth"


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

    def predict(
        self,
        frames: List[np.ndarray],
        input_mask: np.ndarray,
    ) -> Generator[np.ndarray, None, None]:
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
        input_mask = torch.nn.functional.interpolate(
            input_mask, (resized_height, resized_width), mode="nearest"
        )
        input_mask = input_mask.squeeze().numpy()
        # track input objects' masks
        with torch.cuda.amp.autocast(enabled=True):
            for i, frame in enumerate(frames):
                # preprocess frame
                frame = frame.transpose(2, 0, 1)
                frame = torch.from_numpy(frame)
                frame = torch.unsqueeze(frame, 0)
                frame = torch.nn.functional.interpolate(
                    frame, (resized_height, resized_width), mode="nearest"
                )
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
                prediction = torch.nn.functional.interpolate(
                    prediction, (original_height, original_width), mode="nearest"
                )
                prediction = prediction.squeeze().numpy()
                # update progress bar
                if hasattr(self, "video_interface") and self.video_interface is not None:
                    self.video_interface._notify(task="mask tracking")
                # save predicted mask
                yield prediction


model = XMemTracker()
model.serve()
