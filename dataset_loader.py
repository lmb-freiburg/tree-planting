from __future__ import annotations

import glob
import itertools
import logging
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

import utilities


class BaseDataset(Dataset):
    def __init__(
        self,
        random: bool = False,
        crop: int | tuple[int, int] | bool = 256,
        crop_based_on_mask: bool = False,
    ):
        self.random = random
        if crop:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(
                        crop, pad_if_needed=True, padding_mode="edge"
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose([])
        # self.transform = transforms.Compose([transforms.Resize((256, 256), antialias=True)])
        self.data: list = []
        self.crop = crop
        self.crop_based_on_mask = crop_based_on_mask

    def __len__(self):
        return len(self.data)

    def add_normalizations(self, mean, std):
        self.transform.transforms.append(transforms.Normalize(mean, std))


class TmrtDataset(BaseDataset):
    spatial_indices_wo_aveg = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 18, 20]

    def __init__(
        self,
        data_path: str,
        areas: list[str],
        random: bool = False,
        crop: int = 256,
        # image_size: int = 500,
        ignore_temporal_keys: list = None,
        return_identifier: bool = False,
        return_building_mask: bool = True,
        without_aveg: bool = False,
        learn_aggregated: bool = False,
        aggregated_experiment: str = "",
    ):
        super().__init__(random, crop)
        self.spatial_meta_data_path = os.path.join(data_path, "input/spatial_meta_data")
        self.spatial_masks_path = os.path.join(data_path, "input/spatial_masks")
        self.temporal_meta_data_path = os.path.join(data_path, "input/temporal_meta_data")
        self.output_path = os.path.join(data_path, "output")
        self.return_identifier = return_identifier
        self.return_building_mask = return_building_mask
        self.without_aveg = without_aveg
        self.learn_aggregated = learn_aggregated
        self.aggregated_experiment = aggregated_experiment

        ignore_temporal_keys = (
            [] if ignore_temporal_keys is None else ignore_temporal_keys
        )
        self.temporal_keys = [
            key for key in utilities.TEMPORAL_KEYS if key not in ignore_temporal_keys
        ]

        self.load_data(areas, ignore_temporal_keys)

    def load_data(self, areas: list[str], ignore_temporal_keys: list[str] | None = None):
        if ignore_temporal_keys is None:
            ignore_temporal_keys = []

        temporal_meta_data = {
            os.path.basename(day).replace(".csv", ""): pd.read_csv(day).to_dict()
            for day in glob.glob(self.temporal_meta_data_path + "/*.csv")
        }

        for area in areas:
            spatial_meta_data = os.path.join(
                self.spatial_meta_data_path, f"{area}.npy"
            )
            if not os.path.isfile(spatial_meta_data):
                if self.random:
                    raise Exception("Spatial information for area {area} is missing")
                print(f"Area {area} is ignored")
                continue
            mask_data = os.path.join(self.spatial_masks_path, f"{area}.npy")
            if not os.path.isfile(mask_data):
                if self.random:
                    raise Exception("Building mask for area {area} is missing")
                print(f"Area {area} is ignored")
                continue

            if self.learn_aggregated:
                file = os.path.join(self.output_path, area, f"{self.aggregated_experiment}.npy")
                assert os.path.isfile(file)
                self.data.append(
                    {
                        "tmrt": torch.from_numpy(np.load(file)).unsqueeze(0),
                        "spatial_meta_data": torch.from_numpy(np.load(spatial_meta_data)),
                        "spatial_mask": torch.from_numpy(1-np.load(mask_data)).unsqueeze(0),
                        "identifier": f"{area}",
                    }
                )
            else:
                area_days = glob.glob(os.path.join(self.output_path, area) + "/*/")
                for day in area_days:
                    files = sorted(
                        x for x in glob.glob(day + "/*.npy") if "average" not in x
                    )

                    day_dict = temporal_meta_data[day.split("/")[-2]]
                    assert len(day_dict["dt"].keys()) == len(
                        files
                    ), f"Missing data for area {area} for day {day.split('/')[-2]}"
                    for index, file in enumerate(files):
                        self.data.append(
                            {
                                "tmrt": file,
                                "spatial_meta_data": spatial_meta_data,
                                "spatial_mask": mask_data,
                                "temporal_meta_data": utilities.process_temporal_meta_data(
                                    day_dict,
                                    index,
                                    ignored_keys=ignore_temporal_keys,
                                ),
                                "identifier": f"{area}_{day_dict['dt'][index].replace(' ', '_')}",
                            }
                        )

    def __getitem__(self, idx: int | torch.tensor):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # type: ignore[union-attr]

        # load data from disk
        if self.learn_aggregated:
            tmrt = self.data[idx]["tmrt"].clone().detach()
        else:
            tmrt = torch.from_numpy(
                np.expand_dims(np.load(self.data[idx]["tmrt"]), axis=0)
            )
        if self.learn_aggregated:
            spatial_meta = self.data[idx]["spatial_meta_data"].clone().detach()
        else:
            spatial_meta = torch.from_numpy(np.load(self.data[idx]["spatial_meta_data"]))
        if self.without_aveg:
            spatial_meta = spatial_meta[self.spatial_indices_wo_aveg]

        if self.return_building_mask and self.data[idx]["spatial_mask"] is not None:
            if self.learn_aggregated:
                spatial_mask = self.data[idx]["spatial_mask"]
            else:
                spatial_mask_np = 1 - np.load(
                    self.data[idx]["spatial_mask"]
                )  # s.t. 1 is no building, 0 is building
                spatial_mask = torch.from_numpy(spatial_mask_np)[None, ...]
        else:
            spatial_mask = torch.ones_like(tmrt)
        
        if not self.learn_aggregated:
            temporal_meta_t = (
                torch.tensor(self.data[idx]["temporal_meta_data"])
                .type("torch.FloatTensor")
                .unsqueeze(0)
            )

        # transform data
        combined = torch.cat(
            [tmrt, spatial_mask, spatial_meta]
            if self.return_building_mask
            else [tmrt, spatial_meta],
            dim=0,
        ).type("torch.FloatTensor")
        if self.random:
            combined_cropped = self.transform(combined).unsqueeze(0)
        else:
            if self.crop:
                combined_cropped = torch.nn.functional.interpolate(
                    combined.unsqueeze(0), self.crop
                )
            else:
                combined_cropped = combined.unsqueeze(0)

        tmrt_t = combined_cropped[:, 0]
        if self.return_building_mask:
            spatial_mask_t = combined_cropped[:, 1]
            spatial_mask_t = spatial_mask_t.type(torch.int32)
            spatial_meta_t = combined_cropped[:, 2:]
        else:
            spatial_meta_t = combined_cropped[:, 1:]

        tmrt_t[torch.isnan(tmrt_t)] = 0
        if not self.learn_aggregated:
            temporal_meta_t[torch.isnan(temporal_meta_t)] = 0
            temporal_meta_t = temporal_meta_t.squeeze()
        spatial_meta_t[torch.isnan(spatial_meta_t)] = 0
        spatial_meta_t = spatial_meta_t.squeeze()


        if self.learn_aggregated:
            ret_val = (spatial_meta_t, tmrt_t)
        else:
            ret_val = (spatial_meta_t, temporal_meta_t, tmrt_t)
        if self.return_building_mask:
            ret_val += (spatial_mask_t,)  # type: ignore[assignment]
        if self.return_identifier:
            ret_val += (self.data[idx]["identifier"],)  # type: ignore[assignment]

        return ret_val


class DSMV2SVFDataset(Dataset):
    AREA_DIM = 500
    CROP_DIM = 256

    def __init__(
        self,
        vegetation: torch.Tensor,
        svfs: torch.Tensor,
        nodatavals: tuple,
        x_shift: int,
        y_shift: int,
        train: bool = False,
        include_areas: list | None = None,
        exclude_areas: list | None = None,
    ) -> None:
        super().__init__()
        self.data: list = []

        assert (train and include_areas is None) or (
            not train and include_areas is not None
        )
        self.train = train
        self.include_areas = include_areas
        self.exclude_areas = exclude_areas

        if not self.train:
            for y_bottom, x_left in self.include_areas:  # type: ignore[union-attr]
                self.data.append(
                    {
                        "vegetation": self._crop_area(
                            vegetation,
                            y_bottom - y_shift,
                            x_left - x_shift,
                            self.AREA_DIM,
                        ),
                        "svfs": self._crop_area(
                            svfs, y_bottom - y_shift, x_left - x_shift, self.AREA_DIM
                        ),
                    }
                )
        else:
            self.train = True
            self.vegetation = vegetation
            self.svfs = svfs
            ydim, xdim = vegetation.shape
            possible_corners = []
            for y in range(0, ydim - self.CROP_DIM, self.AREA_DIM // 10):
                for x in range(0, xdim - self.CROP_DIM, self.AREA_DIM // 10):
                    if vegetation[y, x] in nodatavals:
                        continue
                    if any(
                        nodataval
                        in vegetation[y : y + self.AREA_DIM, x : x + self.AREA_DIM]
                        for nodataval in nodatavals
                    ):
                        continue
                    possible_corners.append((y, x))
            not_allowed_corners = [
                (y_ - y_shift, x_ - x_shift)
                for y, x in self.exclude_areas  # type: ignore[union-attr]
                for y_ in range(y, y + self.AREA_DIM)
                for x_ in range(x, x + self.AREA_DIM)
            ]
            self.data = list(set(possible_corners) - set(not_allowed_corners))

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _crop_area(entire_area: torch.Tensor, y_bottom: int, x_left: int, crop_size: int):
        if entire_area.dim() == 2:
            return entire_area[
                y_bottom : min(y_bottom + crop_size, entire_area.size(-2) - 1),
                x_left : min(x_left + crop_size, entire_area.size(-1) - 1),
            ]
        elif entire_area.dim() > 2:
            return entire_area[
                ...,
                y_bottom : min(y_bottom + crop_size, entire_area.size(-2) - 1),
                x_left : min(x_left + crop_size, entire_area.size(-1) - 1),
            ]
        raise NotImplementedError

    def random_crop(self, entire_area: torch.Tensor, crop_size: int):
        y, x = random.choice(self.random_corners)
        return self._crop_area(entire_area, y, x, crop_size)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()  # type: ignore[union-attr]

        if self.train:
            vegetation = self.vegetation.unsqueeze(0)
            svfs = self.svfs
            y, x = self.data[index]
        else:
            vegetation = self.data[index]["vegetation"].unsqueeze(0)
            svfs = self.data[index]["svfs"]

        if self.train:
            return self._crop_area(vegetation, y, x, self.CROP_DIM), self._crop_area(
                svfs, y, x, self.CROP_DIM
            )
        return vegetation, svfs
