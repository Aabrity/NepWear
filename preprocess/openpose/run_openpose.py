import pdb

import config
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
# PROJECT_ROOT = Path(__file__).absolute().parents[2]
# PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# sys.path.insert(0, str(PROJECT_ROOT))
import os

import cv2
import einops
import numpy as np
import random
import time
import json

# from pytorch_lightning import seed_everything
from preprocess.openpose.annotator.util import resize_image, HWC3
# from preprocess.openpose.annotator.util import resize_image, HWC3
from preprocess.openpose.annotator.openpose import OpenposeDetector

import argparse
from PIL import Image
import torch
import pdb

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

class OpenPose:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        self.preprocessor = OpenposeDetector()

    def __call__(self, input_image, resolution=384):
        torch.cuda.set_device(self.gpu_id)
        if isinstance(input_image, Image.Image):
            input_image = np.asarray(input_image)
        elif type(input_image) == str:
            input_image = np.asarray(Image.open(input_image))
        else:
            raise ValueError
        with torch.no_grad():
            input_image = HWC3(input_image)
            input_image = resize_image(input_image, resolution)
            H, W, C = input_image.shape
            assert (H == 512 and W == 384), 'Incorrect input image shape'
            pose, detected_map = self.preprocessor(input_image)

            candidate = pose['bodies']['candidate']
            subset = pose['bodies']['subset'][0][:18]
            for i in range(18):
                if subset[i] == -1:
                    candidate.insert(i, [0, 0])
                    for j in range(i, 18):
                        if(subset[j]) != -1:
                            subset[j] += 1
                elif subset[i] != i:
                    candidate.pop(i)
                    for j in range(i, 18):
                        if(subset[j]) != -1:
                            subset[j] -= 1

            candidate = candidate[:18]

            for i in range(18):
                candidate[i][0] *= 384
                candidate[i][1] *= 512

            keypoints = {"pose_keypoints_2d": candidate}
            # with open("/home/aigc/ProjectVTON/OpenPose/keypoints/keypoints.json", "w") as f:
            #     json.dump(keypoints, f)
            # #
            # print(candidate)
            # output_image = cv2.resize(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB), (768, 1024))
            # cv2.imwrite('/home/aigc/ProjectVTON/OpenPose/keypoints/out_pose.jpg', output_image)

        return keypoints


if __name__ == '__main__':

    model = OpenPose()
    model('./images/bad_model.jpg')
# import torch
# import numpy as np
# from PIL import Image

# from pathlib import Path
# import sys
# import os

# PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
# sys.path.insert(0, str(PROJECT_ROOT))

# from preprocess.openpose.annotator.util import resize_image, HWC3
# from preprocess.openpose.annotator.openpose import OpenposeDetector


# class OpenPose:
#     def __init__(self, gpu_id=0):
#         self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
#         self.preprocessor = OpenposeDetector()

#     def __call__(self, input_image, resolution=384):
#         # Load image
#         if isinstance(input_image, Image.Image):
#             input_image = np.array(input_image)
#         elif isinstance(input_image, str):
#             input_image = np.array(Image.open(input_image))
#         else:
#             raise ValueError("Input must be a PIL Image or a file path")

#         # Preprocess image
#         input_image = HWC3(input_image)
#         input_image = resize_image(input_image, resolution)
#         H, W, _ = input_image.shape
#         assert (H, W) == (512, 384), f"Image must be resized to 512x384, got {(H, W)}"

#         with torch.no_grad():
#             pose_result, _ = self.preprocessor(input_image)

#             candidate = pose_result['bodies']['candidate']
#             subset = pose_result['bodies']['subset'][0][:18]  # 18 keypoints

#             # Fix missing keypoints
#             for i in range(18):
#                 if subset[i] == -1:
#                     candidate.insert(i, [0, 0])
#                     for j in range(i, 18):
#                         if subset[j] != -1:
#                             subset[j] += 1
#                 elif subset[i] != i:
#                     candidate.pop(i)
#                     for j in range(i, 18):
#                         if subset[j] != -1:
#                             subset[j] -= 1

#             candidate = candidate[:18]

#             # Scale keypoints to original resolution
#             for i in range(18):
#                 candidate[i][0] *= 384
#                 candidate[i][1] *= 512

#             keypoints = {"pose_keypoints_2d": candidate}

#         return keypoints


# # Optional test run
# if __name__ == "__main__":
#     model = OpenPose(gpu_id=0)  # Set to 0 or -1 to test with/without GPU
#     result = model("images/sample.jpg")  # Replace with your image path
#     print(result)
