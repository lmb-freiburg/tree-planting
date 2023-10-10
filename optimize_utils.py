from copy import deepcopy
import torch
from torch import nn
import argparse
import json
from typing import Tuple
import numpy as np
import os

import network
import utilities

DSM_VEG_INDEX = 2
LCC_INDEX = 3

def load_svf_tmrt_model(tmrt_model_path, dsmv2svf_model_path):
    if os.path.isfile(tmrt_model_path):
        tmrt_model_path = os.path.dirname(tmrt_model_path)
    if os.path.isfile(dsmv2svf_model_path):
        dsmv2svf_model_path = os.path.dirname(dsmv2svf_model_path)
    # load tmrt model
    with open(tmrt_model_path / "args.json", encoding="utf-8") as json_data:
        tmrt_model_args = argparse.Namespace()
        tmrt_model_args.__dict__.update(json.load(json_data))
        tmrt_model = network.ConvEncoderDecoder(tmrt_model_args)
        if "data_parallel" in tmrt_model_args and tmrt_model_args.data_parallel:
            tmrt_model = torch.nn.DataParallel(tmrt_model)
        tmrt_model.load_state_dict(
            torch.load(tmrt_model_path / "model.pth", map_location=torch.device("cpu"))
        )
        tmrt_model = tmrt_model.module
        for param in tmrt_model.parameters():
            param.requires_grad = False
        tmrt_model.eval()

    # load svf model
    with open(dsmv2svf_model_path / "args.json", encoding="utf-8") as json_data:
        svf_model_args = argparse.Namespace()
        svf_model_args.__dict__.update(json.load(json_data))
    svf_model = network.ConvEncoderDecoder(svf_model_args)
    if "data_parallel" in svf_model_args and svf_model_args.data_parallel:
        svf_model = torch.nn.DataParallel(svf_model)
    svf_model.load_state_dict(
        torch.load(dsmv2svf_model_path / "model.pth", map_location=torch.device("cpu"))
    )
    if "data_parallel" in svf_model_args and svf_model_args.data_parallel:
        model = svf_model.module
    svf_model.eval()

    model = network.JointSVFAndTmrtModel(svf_model=svf_model, tmrt_model=tmrt_model)
    return model

def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)

def forward_without_svf(model, spatial, temporal=None, statistics=None):
    padder = utilities.InputPadder(spatial.shape)
    spatial = padder.pad(deepcopy(spatial))
    if temporal is None:
        if len(spatial.shape) == 3:
            spatial = spatial.unsqueeze(0)
        outputs = model.forward_tmrt(spatial, temporal, statistics=statistics)
    else:
        batch_size = temporal.size(0)
        stacked_spatial = torch.stack([spatial for _ in range(batch_size)], dim=0)
        outputs = model.forward_tmrt(stacked_spatial, temporal, statistics=statistics)
    outputs = padder.unpad(outputs)
    return outputs

def forward(model, spatial, temporal=None, statistics=None, return_dsm_veg: bool = False):
    batch_size = 1 if temporal is None else temporal.size(0)
    padder = utilities.InputPadder(spatial.shape)
    spatial = padder.pad(deepcopy(spatial))
    dsm_veg = spatial[DSM_VEG_INDEX].unsqueeze(0)
    dsm_veg = torch.stack([deepcopy(dsm_veg.detach()) for _ in range(batch_size)], dim=0)
    dsm_veg = torch.autograd.Variable(dsm_veg, requires_grad=True)
    dsm_veg.requires_grad_()
    stacked_spatial = torch.stack([spatial for _ in range(batch_size)], dim=0)
    stacked_spatial[:, DSM_VEG_INDEX, None] = dsm_veg
    outputs = model(dsm_veg, stacked_spatial, temporal, statistics)
    outputs = padder.unpad(outputs)
    if return_dsm_veg:
        return outputs, dsm_veg
    return outputs

def calculate_sphere_height(x, y, radius):
    if radius**2 - x**2 - y**2 >= 0:
        return np.sqrt(radius**2 - x**2 - y**2)
    return -1

def compute_tree(crown_diameter: int, tree_height: int) -> torch.Tensor:
    sphere_heights = torch.zeros((crown_diameter, crown_diameter))
    sphere_radius = crown_diameter / 2
    for x_coord in range(-crown_diameter//2,crown_diameter//2+1):
        for y_coord in range(-crown_diameter//2,crown_diameter//2+1):
            sphere_heights[x_coord+crown_diameter//2, y_coord+crown_diameter//2] = calculate_sphere_height(x_coord, y_coord, sphere_radius)
    sphere_heights += tree_height - crown_diameter/2
    sphere_heights[sphere_heights<tree_height-crown_diameter/2] = 0
    return sphere_heights

def idx_not_in_list(full_list: list, potential_new_pair: torch.Tensor, crown_diameter: int) -> bool:
    if len(full_list) == 0:
        return True
    for pair in full_list:
        if potential_new_pair[0].item() in list(range(pair[0].item()-crown_diameter//2, pair[0].item()+crown_diameter//2+1)) and potential_new_pair[1].item() in list(range(pair[1].item()-crown_diameter//2, pair[1].item()+crown_diameter//2+1)):
            return False
    return True

@torch.no_grad()
def plant_trees(
    solution: torch.Tensor,
    spatial_meta: torch.Tensor,
    model: nn.Module,
    tree,
    lcc: bool = True,
) -> torch.Tensor:
    new_spatial_meta = torch.clone(spatial_meta)
    dsm_veg = new_spatial_meta[DSM_VEG_INDEX].unsqueeze(0).unsqueeze(0)
    if isinstance(tree, list):
        for y, x, tree_idx in solution:
            y, x, tree_idx = y.item(), x.item(), tree_idx.item()
            t = tree[tree_idx]
            y1, x1 = t.size(0) // 2, t.size(1) // 2
            y2, x2 = t.size(0) - y1, t.size(1) - x1
            dsm_veg[..., y-y1: y+y2, x-x1:x+x2] = t
            if lcc:
                new_spatial_meta[LCC_INDEX, y-y1: y+y2, x-x1:x+x2] = 5  # grass
    else:
        for y, x in solution:
            y, x = y.item(), x.item()
            dsm_veg[..., y - tree.shape[0]//2 : y + tree.shape[0]//2 + 1, x - tree.shape[1]//2 : x + tree.shape[1]//2 + 1] = tree
            if lcc:
                new_spatial_meta[
                    LCC_INDEX, y - tree.shape[0]//2 : y + tree.shape[0]//2 + 1, x - tree.shape[1]//2 : x + tree.shape[1]//2 + 1
                ] = 5  # grass
    padder = utilities.InputPadder(dsm_veg.shape)
    dsm_veg = padder.pad(deepcopy(dsm_veg))
    new_svf_veg = model.forward_veg_to_svf(dsm_veg)
    new_svf_veg = padder.unpad(new_svf_veg)
    new_spatial_meta[model.svf_indices] = new_svf_veg
    return new_spatial_meta


def gaussian_kernel(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def compute_mean_tmrt(all_outputs: torch.Tensor, mask: torch.Tensor) -> float:
    return torch.mean(all_outputs[:, mask]).item()