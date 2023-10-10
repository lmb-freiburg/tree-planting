import argparse
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm, trange
from copy import deepcopy

from optimize_utils import load_svf_tmrt_model
import utilities
import time

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="datasets/tmrt", type=str)
parser.add_argument("--tmrt_model_path", default="results/tmrt_model/model.pth", type=str)
parser.add_argument("--dsmv2svf_model_path", default="results/dsmv_to_svf/model.pth", type=str)
parser.add_argument("--era5_data_path", default="datasets/era5.csv", type=str)
parser.add_argument("--time_period", default="hottest_day_2020", type=str, choices=["hottest_day_2020", "hottest_week_2020", "year_2020", "decade_2011_2020"])
parser.add_argument("--aggregation", default="mean")
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--area_idx", default=-1, type=int, choices=list(range(60)))

args = parser.parse_args()

start = time.time()

args.data_path = Path(args.data_path)
args.tmrt_model_path = Path(args.tmrt_model_path)
args.dsmv2svf_model_path = Path(args.dsmv2svf_model_path)

device = utilities.get_device()
model = load_svf_tmrt_model(tmrt_model_path=args.tmrt_model_path, dsmv2svf_model_path=args.dsmv2svf_model_path)
model = model.to(device).eval()

areas = utilities.TRAIN_AREAS + utilities.TEST_AREAS
if args.area_idx >= 0:
    areas = areas[args.area_idx]

input_temporal_t = utilities.get_era5_data(era5_data_path=args.era5_data_path, time_period=args.time_period).to(device)

psum = 0
psum_sq = 0
num_pixels = 0
with torch.no_grad():
    for area in tqdm(areas, leave=False, total=len(areas)):
        spatial_meta_path = args.data_path / "input" / "spatial_meta_data" / (area + ".npy")
        spatial_meta = torch.from_numpy(np.load(spatial_meta_path)).float()
        if model.tmrt_model.args.input_channels == 16:
            indices = [0,1,2,3,4,5,6,7,9,10,12,13,15,16,18,20]
            spatial_meta = spatial_meta[indices]
        spatial_meta[torch.isnan(spatial_meta)] = 0
        spatial_meta = spatial_meta.squeeze()
        spatial_meta = spatial_meta.to(device)

        padder = utilities.InputPadder(spatial_meta.shape)
        spatial = padder.pad(deepcopy(spatial_meta))
        stacked_spatial = torch.stack([spatial for _ in range(args.batch_size)], dim=0)
        all_outputs = []
        for outer in trange(0, input_temporal_t.size(0), args.batch_size, leave=False):
            upper = min(outer+args.batch_size, input_temporal_t.size(0))
            outputs = model.forward_tmrt(stacked_spatial[:upper-outer], input_temporal_t[outer:upper], statistics=utilities.STATISTICS["Tmrt"])
            outputs = padder.unpad(outputs)
            all_outputs.append(outputs.detach().cpu())

        all_outputs = torch.concat(all_outputs, dim=0).squeeze()
        if args.aggregation == "mean":
            aggregated_output = all_outputs.mean(dim=0)

        folder = args.data_path / "output" / area
        folder.mkdir(exist_ok=True)
        np.save(folder / f"{args.time_period}.npy", aggregated_output.numpy())

        if area in utilities.TRAIN_AREAS:
            psum += aggregated_output.sum().item()
            psum_sq += (aggregated_output**2).sum().item()
            num_pixels += np.prod(aggregated_output.shape)

        del aggregated_output

mean = psum / num_pixels
var = psum_sq / num_pixels - mean**2
std = np.sqrt(var)

print("Mean", mean)
print("Var", var)
print("Std", std)

end = time.time()
print("Compute time", end-start)