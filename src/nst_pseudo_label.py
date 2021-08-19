"""
This script takes ideas from Noisy Student Training (NST) [1] and
AdaMatch [2] to perform semi-supervision. Even though this is not a
winning solution this approach still yields good results and does not
involve cyclic pseudo-labeling.

[1] https://arxiv.org/abs/1911.04252
[2] https://arxiv.org/abs/2106.04732
"""

import argparse
import numpy as np
import albumentations as A

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import segmentation_models_pytorch as smp
import torch

from etci_dataset import ETCIDataset
from src.utils.nst_utils import soft_distillation_loss, generate_pseudo_labels
from utils import dataset_utils
from utils import metric_utils
from utils import worker_utils
import config

import warnings

warnings.filterwarnings("ignore")

# fix all the seeds and disable non-deterministic CUDA backends for
# reproducibility
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# set up logging
import logging

logging.basicConfig(
    filename="pseudo_label_nst.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def get_dataloader(rank, world_size):
    # create dataframes
    train_df = dataset_utils.create_df(config.train_dir)
    valid_df = dataset_utils.create_df(config.valid_dir)
    train_df = train_df.append(valid_df)
    test_df = dataset_utils.create_df(config.test_dir, "test")

    # filter invalid images
    remove_indices = dataset_utils.filter_df(train_df)
    train_df = train_df.drop(train_df.index[remove_indices])

    # define two sets of augmentation transforms: weak and strong
    weak_transform = A.Compose([A.HorizontalFlip(p=0.5), A.Rotate(270)])
    strong_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(270),
            A.ElasticTransform(
                p=0.4, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
            ),
            A.GridDistortion(p=0.4),
            A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.4),
        ]
    )

    # dataset with weak augmentation
    weak_train_dataset = ETCIDataset(train_df, split="train", transform=weak_transform)
    weak_test_dataset = ETCIDataset(test_df, split="test", transform=weak_transform)
    weak_train_sampler = DistributedSampler(
        weak_train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=2
    )

    # maintain 1:2 ratio for labeled and unlabeled data (idea comes from
    # NST)
    weak_train_loader = DataLoader(
        weak_train_dataset,
        batch_size=36,
        sampler=weak_train_sampler,
        pin_memory=True,
        num_workers=2,
        worker_init_fn=worker_utils.seed_worker,
    )
    weak_test_sampler = DistributedSampler(
        weak_test_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=42
    )
    weak_test_loader = DataLoader(
        weak_test_dataset,
        batch_size=72,
        sampler=weak_test_sampler,
        pin_memory=True,
        num_workers=2,
    )

    # dataset with stronger augmentation
    strong_train_dataset = ETCIDataset(
        train_df, split="train", transform=strong_transform
    )
    strong_test_dataset = ETCIDataset(test_df, split="test", transform=strong_transform)
    strong_train_sampler = DistributedSampler(
        strong_train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=2
    )
    strong_train_loader = DataLoader(
        strong_train_dataset,
        batch_size=36,
        sampler=strong_train_sampler,
        pin_memory=True,
        num_workers=2,
        worker_init_fn=worker_utils.seed_worker,
    )
    strong_test_sampler = DistributedSampler(
        strong_test_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=42
    )
    strong_test_loader = DataLoader(
        strong_test_dataset,
        batch_size=72,
        sampler=strong_test_sampler,
        pin_memory=True,
        num_workers=2,
    )

    return weak_train_loader, weak_test_loader, strong_train_loader, strong_test_loader


def create_student_model():
    model = smp.Unet(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=2,
    )
    return model


def create_teacher_model(type="unet"):
    if type == "unet":
        model = smp.Unet(
            encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=2
        )
    else:
        model = smp.UnetPlusPlus(
            encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=2
        )
    return model


def get_alpha(epoch, total_epochs):
    initial_alpha = 0.1
    final_alpha = 0.5
    modified_alpha = (
        final_alpha - initial_alpha
    ) / total_epochs * epoch + initial_alpha
    return modified_alpha


def train(
    rank, num_epochs, learning_rate, world_size, unet_path, upp_path, student_path
):
    # initialize the workers and fix the seeds
    worker_utils.init_process(rank, world_size)
    torch.manual_seed(0)

    # set up the teacher models and load their checkpoints
    unet_teacher = create_teacher_model("unet")
    upp_teacher = create_teacher_model("upp")

    unet_teacher.load_state_dict(torch.load(unet_path))
    unet_teacher.cuda(rank)
    unet_teacher = DistributedDataParallel(unet_teacher, device_ids=[rank])

    upp_teacher.load_state_dict(torch.load(upp_path))
    upp_teacher.cuda(rank)
    upp_teacher = DistributedDataParallel(upp_teacher, device_ids=[rank])

    teacher_models = [unet_teacher, upp_teacher]

    # student model
    student_model = create_student_model()
    student_model.cuda(rank)
    student_model = DistributedDataParallel(student_model, device_ids=[rank])

    # set up optimizer and LR scheduler
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100], gamma=0.1
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # get the data loaders
    (
        weak_train_loader,
        weak_test_loader,
        strong_train_loader,
        strong_test_loader,
    ) = get_dataloader(rank, world_size)

    # begin training
    for epoch in range(num_epochs):
        losses = metric_utils.AvgMeter()

        if rank == 0:
            print(
                "Rank: {}/{} Epoch: [{}/{}]".format(
                    rank, world_size, epoch + 1, num_epochs
                )
            )

        student_model.train()
        for (
            weak_batch_train,
            weak_batch_test,
            strong_batch_train,
            strong_batch_test,
        ) in zip(
            weak_train_loader, weak_test_loader, strong_train_loader, strong_test_loader
        ):
            with torch.cuda.amp.autocast(enabled=True):
                # load image and mask into device memory
                weak_images_train = weak_batch_train["image"].cuda(
                    rank, non_blocking=True
                )
                weak_images_test = weak_batch_test["image"].cuda(
                    rank, non_blocking=True
                )
                strong_images_train = strong_batch_train["image"].cuda(
                    rank, non_blocking=True
                )
                strong_images_masks = strong_batch_train["mask"].cuda(
                    rank, non_blocking=True
                )
                strong_images_test = strong_batch_test["image"].cuda(
                    rank, non_blocking=True
                )
                num_train = weak_images_train.size(0)

                # pass images into student model
                student_predictions = student_model(
                    torch.cat((strong_images_train, strong_images_test), 0)
                )

                # for filtering the predictions to generate pseudo-labels
                # we make the threshold parameter adaptive
                # (this threshold is ONLY used for unlabeled samples because we consider
                # them to be coming with distribution shifts)
                row_wise_max = torch.nn.Softmax(dim=1)(student_predictions[:num_train])
                row_wise_max = torch.amax(row_wise_max, dim=(1, 2, 3))
                final_sum = row_wise_max.mean(0)
                c_tau = 0.8 * final_sum  # 0.8: initial threshold

                # pass images into teachers
                pseudo_labels, test_mask = generate_pseudo_labels(
                    weak_images_train, weak_images_test, teacher_models, c_tau
                )

                # get loss
                # we use a relative alpha for the training images to
                # adjust the contributions of the labels when present.
                alpha = get_alpha(epoch, num_epochs)
                train_loss = soft_distillation_loss(
                    student_predictions[:num_train],
                    pseudo_labels[:num_train],
                    strong_images_masks,
                    alpha=alpha,
                )
                test_loss = soft_distillation_loss(
                    student_predictions[num_train:],
                    pseudo_labels[num_train:],
                    ground_truths=None,
                    alpha=0.0,
                )
                loss = train_loss + (test_loss[test_mask]).mean()
                losses.update(loss.cpu().item(), (num_train + test_loss.size(0)))

            # update the model
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # update scheduler and average out the losses
        scheduler.step()
        loss = losses.avg
        global_loss = metric_utils.global_meters_all_avg(rank, world_size, loss)

        if rank == 0:
            logging.info(f"Epoch: {epoch+1} Loss: {global_loss[0]:.3f}")

    #  serialization of model weights
    if rank == 0:
        torch.save(student_model.module.state_dict(), f"student_path_{rank}.pth")


WORLD_SIZE = torch.cuda.device_count()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", type=int, default=100, help="number of epochs")
    ap.add_argument("-l", "--lr", type=float, default=1e-3, help="learning rate")
    ap.add_argument(
        "-t1",
        "--teacher-one",
        type=str,
        help="path to teacher one weights",
        default="unet_mobilenet_v2_0.pth",
    )
    ap.add_argument(
        "-t2",
        "--teacher-two",
        type=str,
        help="path to teacher two weights",
        default="upp_mobilenetv2_0.pth",
    )
    ap.add_argument(
        "-s",
        "--student",
        type=str,
        help="path to serialize the student model weights (without .pth)",
        default="nst_student",
    )

    args = vars(ap.parse_args())

    mp.spawn(
        train,
        args=(
            args["epochs"],
            args["lr"],
            WORLD_SIZE,
            args["teacher_one"],
            args["teacher_two"],
            args["student"],
        ),
        nprocs=WORLD_SIZE,
        join=True,
    )
