import argparse
import glob
import json
import os
from contextlib import suppress

import numpy as np
import pandas as pd
import rasterio
import torch
from path import Path
from PIL import Image
from tensorboardX import SummaryWriter
from timm.utils import NativeScaler
from torch import nn, optim
from torch.utils.data import DataLoader

import dataset_loader
import network
import utilities

Image.MAX_IMAGE_PIXELS = 208800000  # 17400*12000


def read_data(data_path: str, without_aveg: bool = False):
    dsm_path = Path(data_path) / "DSM" / "DSM1_25832_Vegetation_0_Lod1OSM.tif"  # type: ignore[operator]
    with Image.open(dsm_path) as vegetation:
        vegetation = np.array(vegetation)
    vegetation = torch.from_numpy(vegetation).type(torch.float32)
    tiff = rasterio.open(dsm_path)
    nodatavals = tiff.nodatavals
    x_shift = int(tiff.bounds.left)
    y_shift = int(tiff.bounds.bottom)

    svfs = []
    for svf_file in sorted(glob.glob(data_path + "/svfs/*veg.tif")):
        if without_aveg and "aveg.tif" in svf_file:
            continue
        with Image.open(svf_file) as svf:
            svfs.append(np.array(svf))
    svfs = torch.from_numpy(np.array(svfs)).type(torch.float32)

    return vegetation, svfs, nodatavals, x_shift, y_shift


parser = argparse.ArgumentParser()
parser.add_argument("exp_path")
parser.add_argument("--data_path", default="datasets/svfs", type=str)
parser.add_argument("--dimension", default=64, type=int)
parser.add_argument("--n_epochs", default=20, type=int)
parser.add_argument("--skip", nargs="*", type=int)
parser.add_argument("--restore_ckpt", action="store_true")
parser.add_argument("--amp", action="store_true")
parser.add_argument("--clip_grad", action="store_true")
parser.add_argument("--without_aveg", action="store_true")
parser.add_argument("--DEBUG", action="store_true")
args = parser.parse_args()

args.exp_path = Path(args.exp_path)
args.exp_path.makedirs_p()

vegetation, svfs, nodatavals, x_shift, y_shift = read_data(
    args.data_path, without_aveg=args.without_aveg
)

args.input_channels = 1
args.global_channels = 0
args.output_channels = len(svfs)

args.learning_rate = 0.001
args.gamma = 0.9999
args.data_parallel = torch.cuda.device_count() > 1
args.batch_size = 32

utilities.set_seed(0)

test_areas = utilities.TEST_AREAS
test_areas = [tuple(map(int, reversed(area.split("_")))) for area in test_areas]
test_data = dataset_loader.DSMV2SVFDataset(
    vegetation=vegetation,
    svfs=svfs,
    nodatavals=nodatavals,
    x_shift=x_shift,
    y_shift=y_shift,
    train=False,
    include_areas=test_areas,
)
test_loader = DataLoader(
    test_data,
    batch_size=len(test_areas),  # only works since test_areas is small (6)
    num_workers=1,
    pin_memory=False,
    shuffle=False,
)

train_data = dataset_loader.DSMV2SVFDataset(
    vegetation=vegetation,
    svfs=svfs,
    nodatavals=nodatavals,
    x_shift=x_shift,
    y_shift=y_shift,
    train=True,
    exclude_areas=test_areas,
)
train_loader = DataLoader(
    train_data,
    batch_size=args.batch_size,
    num_workers=1 if args.DEBUG else 20,
    pin_memory=False,
    shuffle=True,
)

device = utilities.get_device()
model = network.ConvEncoderDecoder(args)
if not args.DEBUG and args.data_parallel:
    model = torch.nn.DataParallel(model)
model.to(device)
criterion = nn.L1Loss()

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
        vegetation, svfs = data_blob
        vegetation = vegetation.to(device)
        svfs = svfs.to(device)

        optimizer.zero_grad()
        with amp_autocast():
            outputs = model(vegetation)
            loss = criterion(outputs, svfs)

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
    model.eval()
    test_errors = []
    with torch.no_grad():
        for data_blob in test_loader:
            vegetation, svfs = data_blob
            vegetation = vegetation.to(device)
            svfs = svfs.to(device)

            padder = utilities.InputPadder(vegetation.shape)
            vegetation = padder.pad(vegetation)
            outputs = model(vegetation)
            outputs = padder.unpad(outputs)
            error = criterion(outputs, svfs)
            test_errors.append(error.item())
    log_writer.add_scalar("val", np.mean(test_errors), epoch)

torch.save(model.state_dict(), args.exp_path / "model.pth")
