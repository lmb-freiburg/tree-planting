import argparse
import json
import os
import random
import time
from contextlib import suppress

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from path import Path
from tensorboardX import SummaryWriter
from timm.utils import NativeScaler
from torch.utils.data import DataLoader

import dataset_loader
import network
import utilities


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path")
    parser.add_argument("--data_path", default="datasets/tmrt", type=str)
    parser.add_argument("--dimension", default=64, type=int)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--skip", nargs="*", type=int)
    parser.add_argument("--ignore_temporal", nargs="*", type=str)
    parser.add_argument("--restore_ckpt", action="store_true")
    parser.add_argument("--apply_mask", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--clip_grad", action="store_true")
    parser.add_argument("--DEBUG", action="store_true")
    parser.add_argument("--without_aveg", action="store_true")
    return parser.parse_args()


args = parse_arguments()

args.input_channels = 16 if args.without_aveg else 21
args.output_channels = 1
args.global_channels = len(utilities.TEMPORAL_KEYS)
if args.ignore_temporal is not None:
    if not all(temp in utilities.TEMPORAL_KEYS for temp in args.ignore_temporal):
        missing_keys = [
            temp for temp in args.ignore_temporal if temp in utilities.TEMPORAL_KEYS
        ]
        raise Exception(
            f"Keys {missing_keys} not in temporal keys {utilities.TEMPORAL_KEYS}"
        )
    args.global_channels -= len(args.ignore_temporal)
args.learning_rate = 0.001
args.gamma = 0.9999
args.data_parallel = torch.cuda.device_count() > 1

args.exp_path = Path(args.exp_path)
args.exp_path.makedirs_p()

utilities.set_seed(0)

test_data = dataset_loader.TmrtDataset(
    args.data_path,
    utilities.TEST_AREAS,
    ignore_temporal_keys=args.ignore_temporal,
    return_building_mask=args.apply_mask,
    without_aveg=args.without_aveg,
)
test_loader = DataLoader(
    test_data,
    batch_size=8,
    num_workers=1 if args.DEBUG else 20,
    pin_memory=False,
)

train_data = dataset_loader.TmrtDataset(
    args.data_path,
    utilities.TRAIN_AREAS,
    random=True,
    ignore_temporal_keys=args.ignore_temporal,
    return_building_mask=args.apply_mask,
    without_aveg=args.without_aveg,
)
train_loader = DataLoader(
    train_data,
    batch_size=32,
    num_workers=1 if args.DEBUG else 20,
    shuffle=True,
    pin_memory=False,
)

device = utilities.get_device()
model = network.ConvEncoderDecoder(args)
if not args.DEBUG and args.data_parallel:
    model = torch.nn.DataParallel(model)
model.to(device)
criterion = utilities.MaskedLoss(
    nn.L1Loss(reduction="none") if args.apply_mask else nn.L1Loss()
)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)

amp_autocast = suppress
loss_scalar = None
if args.amp:
    amp_autocast = torch.cuda.amp.autocast  # type: ignore[misc]
    loss_scaler = NativeScaler()

start_epoch = 1
curr_iter = 0
if args.restore_ckpt:
    if not os.path.isfile(args.exp_path / "checkpoint.pth"):
        print("WARNING: Cannot find checkpoint file -> train from scratch")
    else:
        checkpoint = torch.load(args.exp_path / "checkpoint.pth")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if args.amp:
            loss_scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        curr_iter = checkpoint["curr_iter"]

with open(args.exp_path / "args.json", "w", encoding="utf-8") as f:
    json.dump(args.__dict__, f, indent=2)
log_writer = SummaryWriter(log_dir=args.exp_path)

for epoch in range(start_epoch, args.n_epochs + 1):
    train_loss = 0.0
    model.train()
    for data_blob in train_loader:
        if len(data_blob) == 3:
            spatial_meta, temporal_meta, tmrt = data_blob
            building_mask = None
        elif len(data_blob) == 4 and args.apply_mask:
            spatial_meta, temporal_meta, tmrt, building_mask = data_blob
            building_mask = building_mask.to(device)
        spatial_meta = spatial_meta.to(device)
        temporal_meta = temporal_meta.to(device)
        tmrt = tmrt.to(device)

        optimizer.zero_grad()
        with amp_autocast():
            outputs = model(
                spatial_meta, temporal_meta, statistics=utilities.STATISTICS["Tmrt"]
            )
            loss = criterion(outputs, tmrt, mask=building_mask)

        if loss_scalar is not None:
            # pylint: disable=not-callable
            loss_scalar(loss, optimizer, clip_grad=1.0 if args.clip_grad else None)
            # pylint: enable=not-callable
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        train_loss += loss.item()
        log_writer.add_scalar("loss", loss.item(), curr_iter)
        log_writer.add_scalar("learning_rate", lr_scheduler.get_last_lr()[0], curr_iter)
        curr_iter += 1

    train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch}, Loss: {train_loss:.2f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "scaler_state_dict": loss_scaler.state_dict() if args.amp else None,
            "epoch": epoch,
            "curr_iter": curr_iter,
        },
        args.exp_path / "checkpoint.pth",
        _use_new_zipfile_serialization=False,
    )

    test_error = []
    with torch.no_grad():
        model.eval()
        for data_blob in test_loader:
            if len(data_blob) == 3:
                spatial_meta, temporal_meta, tmrt = data_blob
                building_mask = None
            elif len(data_blob) == 4 and args.apply_mask:
                spatial_meta, temporal_meta, tmrt, building_mask = data_blob
                building_mask = building_mask.to(device)
            spatial_meta = spatial_meta.to(device)
            temporal_meta = temporal_meta.to(device)
            tmrt = tmrt.to(device)
            outputs = model(
                spatial_meta, temporal_meta, statistics=utilities.STATISTICS["Tmrt"]
            )
            error = criterion(outputs, tmrt, mask=building_mask)
            test_error.append(error.item())
    print(f"Epoch {epoch}, Val error: {np.mean(test_error):.2f}")
    log_writer.add_scalar("val", np.mean(test_error), epoch)

torch.save(model.state_dict(), args.exp_path / "model.pth")
