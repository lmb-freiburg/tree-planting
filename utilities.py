from __future__ import annotations

import os
import random
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from path import Path
from PIL import Image
from torchvision.transforms.transforms import _setup_size
from tqdm import tqdm

isiterable = lambda x: isinstance(x, Iterable)

TEST_AREAS = [
    "414000_5318000",
    "413500_5316000",
    "409500_5316500",
    "414000_5320000",
    "414000_5315000",
]

TRAIN_AREAS = [
    '402500_5313500', 
    '402500_5314500', 
    '403500_5316500', 
    '404000_5316500', 
    '404000_5318500', 
    '404000_5319500', 
    '404500_5315500', 
    '405000_5315000', 
    '405000_5315500', 
    '405000_5318000', 
    '406000_5318000', 
    '407000_5315000', 
    '409000_5316000', 
    '409500_5317500', 
    '409500_5318000', 
    '410500_5314000',
    '410500_5315500', 
    '410500_5316500', 
    '410500_5320000', 
    '411000_5314500', 
    '411000_5318500', 
    '411000_5322500', 
    '411500_5314000', 
    '411500_5316000', 
    '411500_5317500', 
    '412000_5318500', 
    '412500_5314500', 
    '412500_5317000', 
    '413000_5315000', 
    '413000_5315500', 
    '413000_5316000', 
    '413000_5316500', 
    '413000_5319500', 
    '413500_5314000', 
    '413500_5314500', 
    '413500_5317500', 
    '413500_5319000', 
    '414000_5313500', 
    '414000_5316000', 
    '414500_5312000', 
    '414500_5314000', 
    '414500_5315500', 
    '414500_5316000', 
    '414500_5317000', 
    '414500_5317500', 
    '414500_5318500', 
    '414500_5319500', 
    '414500_5320000', 
    '415000_5317500', 
    '415500_5314500', 
    '415500_5315000', 
    '416500_5314500', 
    '416500_5315000', 
    '417000_5315000', 
    '417500_5315000', 
    '417500_5316500'
]

assert set(TRAIN_AREAS).intersection(set(TEST_AREAS)) == set()

TEMPORAL_KEYS = [
    "Ta",
    "Wind",
    "Wd",
    "Kdown",
    "rain",
    "RH",
    "press",
    "ElevationAngle",
    "AzimuthAngle",
]


# (mean, std, min, max, out) all values outside min/max (if provided) are set to out
STATISTICS = {
    "r.DEM": (286.96, 108.28),
    "r.DSM.GB": (287.67, 108.13),
    "r.DSM.V": (18.34, 10.61),
    "r.WH": (0.11, 1.01),
    "r.WA": (2.43, 23.53),
    "Tmrt": (22.30, 19.17),
    "aggTmrt_hottest_day_2020": (33.20010692857143, 2.85123240321404),
    "aggTmrt_hottest_week_2020": (26.012906964285715, 1.687793947812605),
    "aggTmrt_year_2020": (12.5912975, 0.9297527217688725),
    "aggTmrt_decade_2011_2020": (11.766492660714286, 0.9240638705118924),
}

def normalize_array(
    x, mean, std, min=None, max=None, replace=None
):  # pylint: disable=redefined-builtin
    if replace:
        return np.where((min <= x) & (x <= max), (x - mean) / std, replace)
    return (x - mean) / std

def denormalize_array(x, mean, std):
    return (x * std) + mean


def get_normalize_func(stats):
    if len(stats) > 2:
        return lambda x: normalize_array(
            x, stats[0], stats[1], stats[2], stats[3], stats[4]
        )
    return lambda x: normalize_array(x, stats[0], stats[1])


def denormalize_multi_dim_array(x, means, stds):
    arrs = []
    for index in range(x.shape[1]):
        arrs.append(denormalize_array(x[:, index], means[index], stds[index]))
    return torch.stack(arrs, dim=1)


def process_temporal_meta_data(
    dict, key, ignored_keys: list = None
):  # pylint: disable=redefined-builtin
    if ignored_keys is None:
        ignored_keys = []
    result = []
    for k in TEMPORAL_KEYS:
        if k in ignored_keys:
            continue
        if k == "dt":
            (h, m, _) = dict[k][key].split(" ")[-1].split(":")
            result.append(int(h) * 60 + int(m))
        else:
            result.append(dict[k][key])
    return result


def load_and_combine_images(img_list, normalize: bool = True):
    arrays = []
    for img_path in img_list:
        if ".tif" == img_path[-4:]:
            with Image.open(img_path) as img:
                array = np.array(img).clip(min=0)
            statistics_name = os.path.basename(img_path).replace(".tif", "")
        elif ".npy" == img_path[-4:]:
            array = np.load(img_path).clip(min=0)
            statistics_name = os.path.basename(img_path).replace(".npy", "")
        else:
            raise NotImplementedError
        if normalize and statistics_name in STATISTICS.keys():
            mean, std = STATISTICS[statistics_name]
            arrays.append(normalize_array(array, mean, std))
        else:
            arrays.append(array)
    return np.stack(arrays, axis=0)


def get_device(debug=False) -> str:
    return "cuda" if torch.cuda.is_available() and not debug else "cpu"


def set_seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    random.seed(seed)


def reshape_output(x, fold, padding):
    batch_size = x.shape[0]
    number_of_crops = x.shape[1]
    x = torch.permute(x, (0, 2, 3, 1)).reshape(batch_size, -1, number_of_crops)
    x = fold(x)
    return x[:, :, padding[0] : -padding[1], padding[2] : -padding[3]]

class InputPadder:
    """Pads images such that dimensions are divisible by factor=2^x"""

    def __init__(self, dims: tuple, factor: int = 8, pad_mode: str = "replicate"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // factor) + 1) * factor - self.ht) % factor
        pad_wd = (((self.wd // factor) + 1) * factor - self.wd) % factor
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, pad_ht // 2, pad_ht - pad_ht // 2]
        self.pad_mode = pad_mode

    def pad(self, x):
        return F.pad(x, self._pad + ([0, 0] if x.dim() > 4 else []), mode=self.pad_mode)

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


class MaskedLoss(torch.nn.modules.loss._Loss):  # pylint: disable=protected-access
    def __init__(self, loss, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average, reduce, reduction)
        self._loss = loss

    def forward(
        self,
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if mask is None:
            return self._loss(input, target)
        loss = self._loss(input, target) * mask
        if self.reduction == "mean":
            if torch.sum(mask) == 0.0:
                return torch.sum(loss)
            return torch.sum(loss) / torch.sum(mask)
        elif self.reduction == "sum":
            return torch.sum(loss)
        return loss

def process_temporal_era5_data(
    dict, ignored_keys: list = None
):  # pylint: disable=redefined-builtin
    if ignored_keys is None:
        ignored_keys = []
    result = []
    for k in TEMPORAL_KEYS:
        if k == "press":
            k = "press_hPa"
        if k in ignored_keys:
            continue
        if k == "dt":
            (h, m, _) = dict[k].split(" ")[-1].split(":")
            result.append(int(h) * 60 + int(m))
        else:
            result.append(dict[k])
    return result

def get_era5_data(era5_data_path, time_period) -> torch.Tensor:
    # load data from era5
    df = pd.read_csv(era5_data_path)
    if time_period == "decade_2011_2020":
        df = df[df["date"].str.contains('|'.join([f"{year}-" for year in [2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011]]))]
        df = df.tail(-1) # drop two times 2011-01-01
    elif "2020" in time_period:
        df = df[df["date"].str.contains("2020-")]
        if time_period == "hottest_day_2020":
            hottest_date = df.groupby(by="date").mean(numeric_only=True)["Ta"].idxmax()
            df = df[df["date"].str.contains(hottest_date)]
        elif time_period == "hottest_week_2020":
            a = list(df.groupby(by="date").max()["Ta"])
            idx = np.argmax([np.mean(a[idx:idx+7]) for idx in range(366-7+1)])
            dates = [date_cet.split(" ")[0] for date_cet in list(df.groupby(by="date").max()["date_CET"])][idx:idx+7]
            df = df[df["date"].str.contains('|'.join(dates))]
        else: # full year 2020
            pass
    else:
        raise NotImplementedError
    
    # preproces data
    input_temporal_t = []
    for _, row in tqdm(df.iterrows(), leave=False, total=len(df)):
        row["press_hPa"] /= 10 # correct to training data
        temporal_meta_t = process_temporal_era5_data(
            row,
            ignored_keys=None,
        )
        temporal_meta_t = torch.tensor(temporal_meta_t).float().unsqueeze(0)
        temporal_meta_t[torch.isnan(temporal_meta_t)] = 0
        input_temporal_t.append(temporal_meta_t)    

    return torch.concat(input_temporal_t, dim=0)