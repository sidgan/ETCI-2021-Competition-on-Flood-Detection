import numpy as np
import albumentations as A

from torch.utils.data import DataLoader
from src.utils import sampler_utils
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.multiprocessing as mp
import segmentation_models_pytorch as smp

from etci_dataset import ETCIDataset
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
    filename="supervised_training.log",
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def get_dataloader(rank, world_size):
    """Creates the data loaders."""
    # create dataframes
    train_df = dataset_utils.create_df(config.train_dir)
    valid_df = dataset_utils.create_df(config.valid_dir)

    # determine if an image has mask or not
    flood_label_paths = train_df["flood_label_path"].values.tolist()
    train_has_masks = list(map(dataset_utils.has_mask, flood_label_paths))
    train_df["has_mask"] = train_has_masks

    # filter invalid images
    remove_indices = dataset_utils.filter_df(train_df)
    train_df = train_df.drop(train_df.index[remove_indices])

    # define augmentation transforms
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(270),
            A.ElasticTransform(
                p=0.4, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
            ),
        ]
    )

    # define datasets
    train_dataset = ETCIDataset(train_df, split="train", transform=transform)
    validation_dataset = ETCIDataset(valid_df, split="validation", transform=None)

    # create samplers
    stratified_sampler = sampler_utils.BalanceClassSampler(
        train_df["has_mask"].values.astype("int")
    )
    train_sampler = sampler_utils.DistributedSamplerWrapper(
        stratified_sampler, rank=rank, num_replicas=world_size, shuffle=True
    )
    val_sampler = DistributedSampler(
        validation_dataset, rank=rank, num_replicas=world_size, shuffle=False
    )

    # create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.local_batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=8,
        worker_init_fn=worker_utils.seed_worker,
    )
    val_loader = DataLoader(
        validation_dataset,
        batch_size=config.local_batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=8,
    )

    return train_loader, val_loader


def create_model(model_class, backbone):
    """Initializes a segmentation model.

    Args:
        model_class: One of smp.Unet, smp.UnetPlusPlus
        backbone: Encoder backbone. Should always be "mobilenet_v2"
                    for this purpose.

    """
    model = model_class(
        encoder_name=backbone, encoder_weights=None, in_channels=3, classes=2
    )
    return model


def train(rank, num_epochs, world_size):
    """Trains the segmentation model using distributed training."""
    # initialize the workers and fix the seeds
    worker_utils.init_process(rank, world_size)
    torch.manual_seed(0)

    # model loading and off-loading to the current device
    model = create_model(smp.Unet, "mobilenet_v2")
    torch.cuda.set_device(rank)
    model.cuda(rank)
    model = DistributedDataParallel(model, device_ids=[rank])

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # set up loss function and gradient scaler for mixed-precision
    criterion_dice = smp.losses.DiceLoss(mode="multiclass")
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # initialize data loaders
    train_loader, val_loader = get_dataloader(rank, world_size)

    ## begin training ##
    for epoch in range(num_epochs):
        losses = metric_utils.AvgMeter()

        if rank == 0:
            logging.info(
                "Rank: {}/{} Epoch: [{}/{}]".format(
                    rank, world_size, epoch + 1, num_epochs
                )
            )

        # train set
        model.train()
        for batch in train_loader:
            with torch.cuda.amp.autocast(enabled=True):
                image = batch["image"].cuda(rank, non_blocking=True)
                mask = batch["mask"].cuda(rank, non_blocking=True)
                pred = model(image)

                loss = criterion_dice(pred, mask)
                losses.update(loss.cpu().item(), image.size(0))

            # update the model
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # average out the losses and log the summary
        loss = losses.avg
        global_loss = metric_utils.global_meters_all_avg(rank, world_size, loss)

        if rank == 0:
            logging.info(f"Epoch: {epoch+1} Train Loss: {global_loss[0]:.3f}")

        ## evaluation ##
        if epoch % 5 == 0:
            if rank == 0:
                logging.info("Running evaluation on the validation set.")
            model.eval()
            losses = metric_utils.AvgMeter()

            with torch.no_grad():
                for batch in val_loader:
                    with torch.cuda.amp.autocast(enabled=True):
                        image = batch["image"].cuda(rank, non_blocking=True)
                        mask = batch["mask"].cuda(rank, non_blocking=True)
                        pred = model(image)

                        loss = criterion_dice(pred, mask)
                        losses.update(loss.cpu().item(), image.size(0))

            loss = losses.avg
            global_loss = metric_utils.global_meters_all_avg(rank, world_size, loss)
            if rank == 0:
                logging.info(f"Epoch: {epoch+1} Val Loss: {global_loss[0]:.3f}")

    # serialization of model weights
    if rank == 0:
        torch.save(
            model.module.state_dict(), f"{config.model_serialization}_{rank}.pth"
        )


WORLD_SIZE = torch.cuda.device_count()

if __name__ == "__main__":
    mp.spawn(train, args=(config.num_epochs, WORLD_SIZE), nprocs=WORLD_SIZE, join=True)
