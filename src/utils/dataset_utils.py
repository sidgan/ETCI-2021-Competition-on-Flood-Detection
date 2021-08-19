"""
Referenced from:
https://medium.com/cloud-to-street/jumpstart-your-machine-learning-satellite-competition-submission-2443b40d0a5a
"""

from glob import glob
import os
import cv2
import numpy as np
import pandas as pd


def has_mask(mask_path):
    img = cv2.imread(mask_path, 0)
    thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]

    if np.mean(thresh) > 0.0:
        return True
    else:
        return False


def get_filename(filepath):
    return os.path.split(filepath)[1]


def create_df(main_dir, split="train"):
    vv_image_paths = sorted(glob(main_dir + "/**/vv/*.png", recursive=True))
    vv_image_names = [get_filename(pth) for pth in vv_image_paths]
    region_name_dates = ["_".join(n.split("_")[:2]) for n in vv_image_names]
    vh_image_paths, flood_label_paths, water_body_label_paths, region_names = (
        [],
        [],
        [],
        [],
    )

    for i in range(len(vv_image_paths)):
        # get vh image path
        vh_image_name = vv_image_names[i].replace("vv", "vh")
        vh_image_path = os.path.join(
            main_dir, region_name_dates[i], "tiles", "vh", vh_image_name
        )
        vh_image_paths.append(vh_image_path)

        # get flood mask path
        if split != "test":
            flood_image_name = vv_image_names[i].replace("_vv", "")
            flood_label_path = os.path.join(
                main_dir, region_name_dates[i], "tiles", "flood_label", flood_image_name
            )
            flood_label_paths.append(flood_label_path)
        elif split == "test":
            flood_label_paths.append(np.NaN)

        # get water body mask path
        water_body_label_name = vv_image_names[i].replace("_vv", "")
        water_body_label_path = os.path.join(
            main_dir,
            region_name_dates[i],
            "tiles",
            "water_body_label",
            water_body_label_name,
        )
        water_body_label_paths.append(water_body_label_path)

        # get region name
        region_name = region_name_dates[i].split("_")[0]
        region_names.append(region_name)

    paths = {
        "vv_image_path": vv_image_paths,
        "vh_image_path": vh_image_paths,
        "flood_label_path": flood_label_paths,
        "water_body_label_path": water_body_label_paths,
        "region": region_names,
    }

    return pd.DataFrame(paths)


def filter_df(df):
    remove_indices = []
    for i, image_path in enumerate(df["vv_image_path"].tolist()):
        # load image
        image = cv2.imread(image_path, 0)

        # get all unique values in image
        image_values = list(np.unique(image))

        # check values
        binary_value_check = (
            (image_values == [0, 255])
            or (image_values == [0])
            or (image_values == [255])
        )

        if binary_value_check is True:
            remove_indices.append(i)
        return remove_indices
