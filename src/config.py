import os
import segmentation_models_pytorch as smp

# dataset related
dataset_root = "/home/jupyter/Flood_Comp/"
train_dir = os.path.join(dataset_root, "train")
valid_dir = os.path.join(dataset_root, "test")
local_batch_size = 96

# model related
backbone = "mobilenet_v2"
model_family = smp.Unet

# training related
learning_rate = 1e-3
model_serialization = "unet_mobilenet_v2"
num_epochs = 15
