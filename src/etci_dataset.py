"""
Referenced from:
https://medium.com/cloud-to-street/jumpstart-your-machine-learning-satellite-competition-submission-2443b40d0a5a
"""

import cv2
import numpy as np

from torch.utils.data import Dataset


def s1_to_rgb(vv_image, vh_image):
    ratio_image = np.clip(np.nan_to_num(vh_image / vv_image, 0), 0, 1)
    rgb_image = np.stack((vv_image, vh_image, 1 - ratio_image), axis=2)
    return rgb_image


class ETCIDataset(Dataset):
    def __init__(self, dataframe, split, transform=None):
        self.split = split
        self.dataset = dataframe
        self.transform = transform

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        example = {}

        df_row = self.dataset.iloc[index]

        # load vv and vh images
        vv_image = cv2.imread(df_row["vv_image_path"], 0) / 255.0
        vh_image = cv2.imread(df_row["vh_image_path"], 0) / 255.0

        # convert vv and vh images to rgb
        rgb_image = s1_to_rgb(vv_image, vh_image)

        if self.split == "test":
            # no flood mask should be available
            example["image"] = rgb_image.transpose((2, 0, 1)).astype("float32")
        else:
            # load ground truth flood mask
            flood_mask = cv2.imread(df_row["flood_label_path"], 0) / 255.0

            # apply augmentations if specified
            if self.transform:
                augmented = self.transform(image=rgb_image, mask=flood_mask)
                rgb_image = augmented["image"]
                flood_mask = augmented["mask"]

            example["image"] = rgb_image.transpose((2, 0, 1)).astype("float32")
            example["mask"] = flood_mask.astype("int64")

        return example
