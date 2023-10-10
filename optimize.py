from __future__ import annotations

from path import Path
import torch
import numpy as np

import utilities
import matplotlib.pyplot as plt
from torch import nn
from copy import deepcopy
import time
from tqdm import trange
import argparse
import os
import json
import pygad
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

from optimize_genetic_utils import create_initial_population, fitness_func, mutation_func, crossover_func
from optimize_utils import plant_trees, load_svf_tmrt_model, unravel_index, forward, LCC_INDEX, DSM_VEG_INDEX, idx_not_in_list, gaussian_kernel, compute_tree, compute_mean_tmrt, forward_without_svf

torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument("--output_path", default="results/optimization", type=str)
parser.add_argument("--data_path", default="datasets/tmrt", type=str)
parser.add_argument("--aggregated_tmrt_model_path", default="results/aggretated_tmrt/hottest_day_2020/model.pth", type=str)
parser.add_argument("--tmrt_model_path", default="results/tmrt_model/model.pth", type=str)
parser.add_argument("--dsmv2svf_model_path", default="results/dsmv_to_svf/model.pth", type=str)
parser.add_argument("--era5_data_path", default="datasets/era5.csv", type=str)
parser.add_argument("--time_period", default="hottest_day_2020", type=str, choices=["hottest_day_2020", "hottest_week_2020", "year_2020", "decade_2011_2020"])
parser.add_argument("--method", default="ìls_genetic_hill_climbing_iterated", type=str, choices=["random", "greedy", "genetic", "ils_genetic_hill_climbing_iterated", "ils_genetic_hill_climbing"])
parser.add_argument("--area", default="413500_5316000", type=str, choices=utilities.TEST_AREAS)
parser.add_argument("--crown_diameter", default=9, type=int, choices=[3,5,7,9,11,13])
parser.add_argument("--ratio", default=2, type=int, choices=[2,3,4]) # urban trees have ratios of 2:1 to 4:1
parser.add_argument("--number_of_trees", default=100, type=int)
parser.add_argument("--neighborhood_size", default=250, type=int)
parser.add_argument("--old_trees", action="store_true")
parser.add_argument("--extract_trees", action="store_true")
parser.add_argument("--with_lcc", action="store_true")

parser.add_argument("--batch_size", default=8, type=int)

parser.add_argument("--plot", action="store_true")
parser.add_argument("--recreate", action="store_true")
parser.add_argument("--save_timesteps", action="store_true")

parser.add_argument("--DEBUG", action="store_true")

args = parser.parse_args()

args.data_path = Path(args.data_path)
args.aggregated_tmrt_model_path = Path(args.aggregated_tmrt_model_path)
args.tmrt_model_path = Path(args.tmrt_model_path)
args.dsmv2svf_model_path = Path(args.dsmv2svf_model_path)
args.output_path = Path(args.output_path)
if args.extract_trees:
    args.output_path = args.output_path / args.time_period / args.area / "extracted"
else:
    args.output_path = args.output_path / args.time_period / args.area / f"{args.crown_diameter}_{args.number_of_trees}_{args.ratio}"
if args.with_lcc:
    args.output_path += "_with_lcc"
args.output_path.makedirs_p()

assert args.crown_diameter%2==1
if args.old_trees:
    args.tree_height = args.crown_diameter*args.ratio
else:
    args.tree_height = args.crown_diameter/3*4 # 25% trunk height
assert 0 < args.number_of_trees < 500*500

if os.path.isfile(args.output_path / f"{args.method}.json"):
    with open(args.output_path / f"{args.method}.json", "r") as f:
        LOG_DICT = json.load(f)
else:
    LOG_DICT = {}

device = utilities.get_device()
if args.DEBUG:
    device = "cuda:1"

LOG_DICT["device"] = torch.cuda.get_device_name()

# load svf & tmrt model
try:
    optimize_model = load_svf_tmrt_model(tmrt_model_path=args.aggregated_tmrt_model_path, dsmv2svf_model_path=args.dsmv2svf_model_path)
    optimize_model = optimize_model.to(device).eval()
except Exception:
    pass
eval_model = load_svf_tmrt_model(tmrt_model_path=args.tmrt_model_path, dsmv2svf_model_path=args.dsmv2svf_model_path)

eval_model = eval_model.to(device).eval()

spatial_meta_path = args.data_path / "input" / "spatial_meta_data" / (args.area + ".npy")
spatial_meta = torch.from_numpy(np.load(spatial_meta_path)).float()
if eval_model.tmrt_model.args.input_channels == 16:
    indices = [0,1,2,3,4,5,6,7,9,10,12,13,15,16,18,20]
    spatial_meta = spatial_meta[indices]
spatial_meta[torch.isnan(spatial_meta)] = 0
spatial_meta = spatial_meta.squeeze()
spatial_meta = spatial_meta.to(device)

# 1 -> paved, 2 -> building, 5 -> grass, 6 -> bare soil, 7 -> water
lcc = spatial_meta[LCC_INDEX]
valid_map = torch.logical_or(lcc == 1, torch.logical_or(lcc == 5, lcc == 6)).to(device) # exclude buildings and water

if args.extract_trees:
    TREE_THRESHOLD = 3.0
    dsm_veg = spatial_meta[DSM_VEG_INDEX].clone().cpu()
    thresholded_dsm_veg = dsm_veg >= TREE_THRESHOLD
    prev_canopy_area = torch.sum(thresholded_dsm_veg).item()

    coords = peak_local_max(dsm_veg.numpy(), footprint=np.ones((3, 3)), labels=thresholded_dsm_veg.numpy())
    mask = np.zeros(dsm_veg.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-dsm_veg.numpy(), markers, mask=thresholded_dsm_veg.numpy())
    unique_labels, counts = np.unique(labels, return_counts=True)
    sorted_unique_labels = [x for _, x in sorted(zip(counts, unique_labels))][::-1][1:]
    sorted_counts = [c for c in sorted(counts)][::-1]
    counts_cumsum = np.cumsum(sorted_counts[1:]) # exlcude background class (0)
    first_larger_index = np.where(counts_cumsum > prev_canopy_area)[0]
    if len(first_larger_index) > 0:
        sorted_unique_labels = sorted_unique_labels[first_larger_index]
    else: # is already smaller
        pass

    extracted_trees = []
    tree_valid_map = valid_map.clone()
    for l in sorted_unique_labels:
        true_indices = np.where(labels == l)
        y1, x1 = np.min(true_indices, axis=1)
        y2, x2 = np.max(true_indices, axis=1)
        y2 += 1
        x2 += 1
        extracted_trees.append(torch.where(torch.from_numpy(labels[y1:y2,x1:x2]==l), dsm_veg[y1:y2, x1:x2], torch.zeros_like(dsm_veg[y1:y2, x1:x2])))

    # compute average tree height & crown diameter
    args.tree_height = np.mean([tree[tree > 0].mean().item() for tree in extracted_trees])
    args.crown_diameter = np.mean([np.mean(list(tree.shape)) for tree in extracted_trees])
    if int(args.crown_diameter) % 2 == 1:
        args.crown_diameter = int(args.crown_diameter)
    else:
        args.crown_diameter = int(args.crown_diameter+1)
    args.number_of_trees = len(extracted_trees)

    np.savez(args.output_path / "extracted_trees.npz", *[t.numpy() for t in extracted_trees])

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=args.crown_diameter, bias=False)
with torch.no_grad():
    # conv.weight = torch.nn.Parameter(torch.ones_like(conv.weight)/math.prod(conv.weight.shape), requires_grad=False)
    conv.weight = torch.nn.Parameter(torch.from_numpy(gaussian_kernel(l=args.crown_diameter)).unsqueeze(0).unsqueeze(0).float(), requires_grad=False)
conv = conv.to(device)

input_temporal_t = utilities.get_era5_data(era5_data_path=args.era5_data_path, time_period=args.time_period).to(device)
args.time_period = "one_day" # TODO

with torch.no_grad():
    convoluted_valid_map = conv(valid_map.unsqueeze(0).float()).squeeze(0) > 0.999999

# record data
if args.recreate or (not os.path.isfile(args.output_path / "tmrt_before.pth")):
    start = time.time()
    save_outputs = [] if input_temporal_t.size(0) < 10000 or args.save_timesteps else torch.zeros((1,) + spatial_meta.shape[1:])
    with torch.no_grad():
        for outer in trange(0, input_temporal_t.size(0), args.batch_size, leave=False):
            outputs = forward(eval_model, spatial_meta, input_temporal_t[outer:outer+args.batch_size], statistics=utilities.STATISTICS["Tmrt"])
            if input_temporal_t.size(0) < 10000 or args.save_timesteps:
                save_outputs.append(outputs.detach().cpu())
            else:
                save_outputs += outputs.detach().cpu().sum(dim=0)
    if input_temporal_t.size(0) < 10000 or args.save_timesteps:
        prev_all_outputs = torch.concat(save_outputs, dim=0).squeeze()
    else:
        prev_all_outputs = save_outputs / input_temporal_t.size(0)

    end = time.time()
    LOG_DICT["pre_optimization_time"] = end - start

    if input_temporal_t.size(0) < 10000 or args.save_timesteps:
        torch.save(prev_all_outputs, args.output_path / "all_outputs.pth")
    prev_mean_tmrt = compute_mean_tmrt(prev_all_outputs, valid_map.cpu())
    LOG_DICT["prev_mean_tmrt"] = prev_mean_tmrt
    torch.save(prev_mean_tmrt, args.output_path / "tmrt_before.pth")

    tmrt_before = prev_all_outputs.mean(dim=0)
    np.save(args.output_path / "tmrt_before.npy", tmrt_before)

    veg_before = spatial_meta[DSM_VEG_INDEX].detach().cpu().numpy()
    np.save(args.output_path / "veg_before.npy", veg_before)

    if args.save_timesteps:
        del prev_all_outputs

    if args.plot:
        plt.imshow(veg_before)
        plt.colorbar()
        plt.savefig(args.output_path / f"veg_before.png")
        plt.close()

        img = np.ma.masked_where(~valid_map.cpu().numpy(), tmrt_before)
        plt.imshow(img)
        plt.colorbar()
        plt.savefig(args.output_path / f"tmrt_before.png")
        plt.close()
else:
    if "decade" not in args.time_period:
        prev_all_outputs = torch.load(args.output_path / "all_outputs.pth")
    prev_mean_tmrt = torch.load(args.output_path / "tmrt_before.pth")
    LOG_DICT["prev_mean_tmrt"] = prev_mean_tmrt
    tmrt_before = torch.from_numpy(np.load(args.output_path / "tmrt_before.npy"))
    veg_before = spatial_meta[DSM_VEG_INDEX].detach().cpu().numpy()

# optimization part
start = time.time()
if args.method == "random":
    all_possible_tree_locations = np.array([[y,x] for y in range(convoluted_valid_map.size(0)) for x in range(convoluted_valid_map.size(1)) if convoluted_valid_map[y,x].item()])
    np.random.shuffle(all_possible_tree_locations)
    final_unraveled_indices = []
    for idx_pair in torch.from_numpy(all_possible_tree_locations):
        if convoluted_valid_map[idx_pair[0], idx_pair[1]] and idx_not_in_list(final_unraveled_indices, idx_pair, args.crown_diameter):
            final_unraveled_indices.append(idx_pair)
        if len(final_unraveled_indices) == args.number_of_trees:
            break
    final_unraveled_indices = torch.stack(final_unraveled_indices,dim=0)
    final_unraveled_indices += args.crown_diameter // 2
elif "greedy" in args.method:
    prev_output = forward_without_svf(optimize_model, spatial_meta, statistics=utilities.STATISTICS[f"aggTmrt_{args.time_period}"]).squeeze()
    indicator = prev_output.squeeze()
    indicator = prev_output[..., args.crown_diameter//2:-args.crown_diameter//2+1, args.crown_diameter//2:-args.crown_diameter//2+1]
    final_unraveled_indices = []
    indices = torch.argsort(indicator.reshape(-1), descending=True)
    for idx_pair in unravel_index(indices, indicator.shape):
        if convoluted_valid_map[idx_pair[0], idx_pair[1]] and idx_not_in_list(final_unraveled_indices, idx_pair, args.crown_diameter):
            final_unraveled_indices.append(idx_pair)
        if len(final_unraveled_indices) == args.number_of_trees:
            break
    final_unraveled_indices = torch.stack(final_unraveled_indices,dim=0)
    final_unraveled_indices += args.crown_diameter // 2
elif "genetic" in args.method and "ils" not in args.method:
    if args.extract_trees:
        all_possible_tree_locations = np.array([[y,x] for y in range(valid_map.size(0)) for x in range(valid_map.size(1)) if valid_map[y,x].item()])
    else:
        all_possible_tree_locations = np.array([[y,x] for y in range(convoluted_valid_map.size(0)) for x in range(convoluted_valid_map.size(1)) if convoluted_valid_map[y,x].item()])
    initial_population = create_initial_population(
        size=20,
        crown_diameter=args.crown_diameter, 
        number_of_trees=args.number_of_trees, 
        valid_map=convoluted_valid_map,
        p=None,
        all_possible_tree_locations=all_possible_tree_locations
    )
    tree = compute_tree(args.crown_diameter, args.tree_height).to(device)
    fitness_func_fn = fitness_func(
        model=eval_model if args.method == "genetic_eval" else optimize_model, 
        tree=extracted_trees if args.extract_trees else tree, 
        spatial_meta=spatial_meta,
        valid_map=valid_map,
        batch_size=args.batch_size,
        crown_diameter=args.crown_diameter,
        statistics=utilities.STATISTICS["Tmrt" if args.method == "genetic_eval" else f"aggTmrt_{args.time_period}"],
        temporal_meta=input_temporal_t if "genetic_eval" in args.method else None,
        lcc=args.with_lcc,
    )
    crossover_func_fn = crossover_func(
        crown_diameter=args.crown_diameter,
        number_of_trees=args.number_of_trees,
        valid_map=valid_map if args.extract_trees else convoluted_valid_map,
        p=None,
        all_possible_tree_locations=all_possible_tree_locations,
        trees=extracted_trees if args.extract_trees else None,
    )
    mutation_func_fn = mutation_func(
        crown_diameter=args.crown_diameter, 
        valid_map=valid_map if args.extract_trees else convoluted_valid_map,
        trees=extracted_trees if args.extract_trees else None,
    )
    ga_instance = pygad.GA(
        num_generations=1000*5, #10000,
        num_parents_mating=2,
        initial_population=initial_population,
        fitness_func=fitness_func_fn,
        crossover_type=crossover_func_fn,
        mutation_type=mutation_func_fn,
        save_best_solutions=True,
    )
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution(
        ga_instance.last_generation_fitness
    )
    print(solution_fitness)
    final_unraveled_indices = torch.from_numpy(solution.reshape(-1, 2)).int()
    final_unraveled_indices += args.crown_diameter // 2
elif "ils" in args.method:
    if args.extract_trees: # remove all vegetation and seal it
        if args.with_lcc:
            spatial_meta[LCC_INDEX][spatial_meta[DSM_VEG_INDEX] > 0] = 1 # paved
        spatial_meta[DSM_VEG_INDEX][spatial_meta[DSM_VEG_INDEX] > 0] = 0 # remove prior vegetation
        # sort trees from largest to smallest
        extracted_trees = [tree.to(device) for _, tree in sorted(zip([tree.shape[0]*tree.shape[1] for tree in extracted_trees], extracted_trees), key=lambda pair: pair[0], reverse=True)]
    with torch.no_grad():
        prev_spatial = plant_trees(
                [], 
                torch.clone(spatial_meta), 
                optimize_model, 
                None,
                lcc=args.with_lcc,
            ).detach()
        prev_output = forward_without_svf(optimize_model, prev_spatial, statistics=utilities.STATISTICS[f"aggTmrt_{args.time_period}"])
    baseline_indicator = prev_output.squeeze()
    baseline_indicator_mean = baseline_indicator[valid_map].mean()
    all_possible_tree_locations = np.array([[y,x] for y in range(convoluted_valid_map.size(0)) for x in range(convoluted_valid_map.size(1)) if convoluted_valid_map[y,x].item()])
    if os.path.isfile(args.output_path / f"delta_tmrt.pth"):
        delta_tmrt = torch.load(args.output_path / f"delta_tmrt.pth")
    else:
        if args.extract_trees:
            tree = compute_tree(args.crown_diameter, args.tree_height).to(device) # compute delta tmrt with average tree
        else:
            tree = compute_tree(args.crown_diameter, args.tree_height).to(device)
        delta_tmrt = torch.zeros(convoluted_valid_map.shape)
        with torch.no_grad():
            for b in trange(0, len(all_possible_tree_locations), args.batch_size, leave=False):
                new_spatial_meta = torch.stack(
                    [plant_trees([all_possible_tree_locations[b_] + args.crown_diameter // 2], torch.clone(spatial_meta), optimize_model, tree, lcc=args.with_lcc).detach() 
                     for b_ in range(b, min(b+args.batch_size, len(all_possible_tree_locations)))], 
                    dim=0
                )
                alternative_indicator = forward_without_svf(optimize_model, new_spatial_meta, statistics=utilities.STATISTICS[f"aggTmrt_{args.time_period}"])
                alternative_indicator_mean = alternative_indicator.squeeze(dim=1)[..., valid_map].mean(dim=(-1))
                for idx, b_ in enumerate(range(b, min(b+args.batch_size, len(all_possible_tree_locations)))):
                    delta_tmrt[all_possible_tree_locations[b_, 0], all_possible_tree_locations[b_, 1]] = baseline_indicator_mean - alternative_indicator_mean[idx]

        torch.save(delta_tmrt, args.output_path / f"delta_tmrt.pth")
        if args.plot:
            img = np.ma.masked_where(~convoluted_valid_map.cpu().numpy(), delta_tmrt.numpy())
            plt.imshow(img)
            plt.colorbar()
            plt.savefig(args.output_path / f"delta_tmrt.jpg")
            plt.close()

    if args.extract_trees:
        # add padding for smaller than average crown diameter trees
        new_delta_tmrt = torch.nn.functional.pad(delta_tmrt.unsqueeze(0).unsqueeze(0), pad=(args.crown_diameter//2, args.crown_diameter//2, args.crown_diameter//2, args.crown_diameter//2), mode="replicate").squeeze()
        delta_tmrt = new_delta_tmrt
        prob_delta_tmrt = delta_tmrt.exp() / delta_tmrt[valid_map.cpu()].exp().sum()
        prob_delta_tmrt[~valid_map.cpu()] = 0
    else:
        prob_delta_tmrt = delta_tmrt.exp() / delta_tmrt[convoluted_valid_map.cpu()].exp().sum()
        prob_delta_tmrt[~convoluted_valid_map.cpu()] = 0

    if args.extract_trees:
        all_possible_tree_locations = np.array([[y,x] for y in range(valid_map.size(0)) for x in range(valid_map.size(1)) if valid_map[y,x].item()])
    
    final_unraveled_indices = []
    if "random_init" in args.method:
        while len(final_unraveled_indices) != args.number_of_trees:
            indices = np.random.choice(np.arange(len(all_possible_tree_locations)), size=200, replace=False, p=prob_delta_tmrt[convoluted_valid_map.cpu()].numpy())
            for idx in indices:
                idx_pair = torch.Tensor(all_possible_tree_locations[idx]).int()
                if convoluted_valid_map[idx_pair[0], idx_pair[1]] and idx_not_in_list(final_unraveled_indices, idx_pair, args.crown_diameter):
                    final_unraveled_indices.append(idx_pair)
                if len(final_unraveled_indices) == args.number_of_trees:
                    break
    else:
        indices = torch.argsort(delta_tmrt.view(-1), descending=True)
        if args.extract_trees:
            tree_convoluted_valid_map = valid_map.clone()
            unraveled_indices = unravel_index(indices, delta_tmrt.shape)
            for tree_idx, tree in enumerate(extracted_trees):
                y1, x1 = tree.size(0) // 2, tree.size(1) // 2
                y2, x2 = tree.size(0) - y1, tree.size(1) - x1
                for idx_pair in unraveled_indices:
                    y, x = idx_pair[0], idx_pair[1]
                    if not tree_convoluted_valid_map[y, x]:
                        continue
                    if y-y1 < 0 or y+y2 > tree_convoluted_valid_map.size(0) or x-x1<0 or x+x2 > tree_convoluted_valid_map.size(0):
                        continue
                    if torch.all(tree_convoluted_valid_map[y-y1: y+y2, x-x1:x+x2]):
                        final_unraveled_indices.append(torch.concat([idx_pair, torch.Tensor([tree_idx])], dim=0).int())
                        tree_convoluted_valid_map[y-y1: y+y2, x-x1:x+x2] = False
                        # remove all indices from unraveled indices that now have been covered by the tree
                        for y_ in range(y-y1, y+y2):
                            for x_ in range(x-x1, x+x2):
                                index_map = torch.logical_and(unraveled_indices[:, 0] == y_, unraveled_indices[:, 1] == x_)
                                assert torch.sum(index_map).item() <= 1
                                if torch.sum(index_map).item() == 1:
                                    i = index_map.nonzero().item()
                                    unraveled_indices = torch.cat([unraveled_indices[0:i], unraveled_indices[i+1:]])
                        break
            assert len(final_unraveled_indices) == len(extracted_trees)
        else:
            for idx_pair in unravel_index(indices, delta_tmrt.shape):
                if convoluted_valid_map[idx_pair[0], idx_pair[1]] and idx_not_in_list(final_unraveled_indices, idx_pair, args.crown_diameter):
                    final_unraveled_indices.append(idx_pair)
                if len(final_unraveled_indices) == args.number_of_trees:
                    break
    final_unraveled_indices = torch.stack(final_unraveled_indices,dim=0)

    if args.extract_trees:
        p = prob_delta_tmrt[valid_map.cpu()].cpu().numpy()
    else:
        p = prob_delta_tmrt[convoluted_valid_map.cpu()].numpy()

    tree = compute_tree(args.crown_diameter, args.tree_height).to(device)
    fitness_func_fn = fitness_func(
        model=eval_model if args.method == "genetic_eval" else optimize_model, 
        tree=extracted_trees if args.extract_trees else tree, 
        spatial_meta=spatial_meta,
        valid_map=valid_map,
        batch_size=args.batch_size,
        crown_diameter=args.crown_diameter,
        statistics=utilities.STATISTICS["Tmrt" if args.method == "genetic_eval" else f"aggTmrt_{args.time_period}"],
        temporal_meta=input_temporal_t if "genetic_eval" in args.method else None,
        lcc=args.with_lcc,
    )
    crossover_func_fn = crossover_func(
        crown_diameter=args.crown_diameter,
        number_of_trees=args.number_of_trees,
        valid_map=valid_map if args.extract_trees else convoluted_valid_map,
        p=p,
        all_possible_tree_locations=all_possible_tree_locations,
        trees=extracted_trees if args.extract_trees else None,
    )
    mutation_func_fn = mutation_func(
        crown_diameter=args.crown_diameter, 
        valid_map=valid_map if args.extract_trees else convoluted_valid_map,
        trees=extracted_trees if args.extract_trees else None,
    )

    cur_best_fitness_list = [fitness_func_fn(final_unraveled_indices.clone().view(-1).numpy(), None)*-1]
    cur_bests = [final_unraveled_indices]
    for iteration in trange(5 if "iterated" in args.method else 1):
        initial_population = [gene.view(-1).numpy() for gene in cur_bests]
        while len(initial_population) < 20:
            gene = []
            complete_gene = False
            while not complete_gene:
                if args.extract_trees:
                    new_tree_valid_map = valid_map.clone()
                    gene = []
                    for tree_idx, t in enumerate(extracted_trees):
                        y1, x1 = t.size(0) // 2, t.size(1) // 2
                        y2, x2 = t.size(0) - y1, t.size(1) - x1
                        gene_indices = np.random.choice(np.arange(len(all_possible_tree_locations)), size=200, replace=False, p=p)
                        for idx in gene_indices:
                            y,x = all_possible_tree_locations[idx]
                            if not new_tree_valid_map[y, x]:
                                continue
                            if y-y1 < 0 or y+y2 > new_tree_valid_map.size(0) or x-x1<0 or x+x2 > new_tree_valid_map.size(0):
                                continue
                            if torch.all(new_tree_valid_map[y-y1: y+y2, x-x1:x+x2]):
                                new_tree_valid_map[y-y1: y+y2, x-x1:x+x2] = False
                                gene.append(torch.Tensor([y, x, tree_idx]).int())
                                break
                    complete_gene = len(gene) == len(extracted_trees)
                else:
                    gene_indices = np.random.choice(np.arange(len(all_possible_tree_locations)), size=200, replace=False, p=p)
                    for idx in gene_indices:
                        idx_pair = torch.Tensor(all_possible_tree_locations[idx]).int()
                        if convoluted_valid_map[idx_pair[0], idx_pair[1]] and idx_not_in_list(gene, idx_pair, args.crown_diameter):
                            gene.append(idx_pair)
                        if len(gene) == args.number_of_trees:
                            complete_gene = True
                            break
            initial_population.append(torch.stack(gene, dim=0).view(-1).numpy())
        
        if "genetic" in args.method:
            ga_instance = pygad.GA(
                num_generations=1000 if "iterated" in args.method else 1000, #10000,
                num_parents_mating=2,
                initial_population=initial_population,
                fitness_func=fitness_func_fn,
                crossover_type=crossover_func_fn,
                mutation_type=mutation_func_fn,
                save_best_solutions=True,
            )
            ga_instance.run()
            solution, solution_fitness, solution_idx = ga_instance.best_solution(
                ga_instance.last_generation_fitness
            )
            if args.extract_trees:
                cur_bests.append(torch.from_numpy(solution.reshape(-1, 3)).int())
            else:
                cur_bests.append(torch.from_numpy(solution.reshape(-1, 2)).int())
            cur_best_fitness_list.append(fitness_func_fn(cur_bests[-1].clone().view(-1).numpy(), None)*-1)
        
        if "hill_climbing" in args.method:
            start_time_hill_climbing = time.time()
            cur_best = deepcopy(cur_bests[-1])
            if "genetic" not in args.method and iteration > 0:
                if args.extract_trees:
                    cur_best = torch.from_numpy(initial_population[len(cur_bests)]).reshape(-1, 3).int()
                else:
                    cur_best = torch.from_numpy(initial_population[len(cur_bests)]).reshape(-1, 2).int()
            cur_best_fitness_prev = fitness_func_fn(cur_best.view(-1).numpy(), None)*-1
            cur_best_fitness = deepcopy(cur_best_fitness_prev)

            if args.extract_trees:
                tree_convoluted_valid_map = valid_map.clone()
                for tree_position_idx in range(args.number_of_trees):
                    tree_position_y, tree_position_x, tree_idx = cur_best[tree_position_idx]
                    t = extracted_trees[tree_idx]
                    y1, x1 = t.size(0) // 2, t.size(1) // 2
                    y2, x2 = t.size(0) - y1, t.size(1) - x1
                    tree_convoluted_valid_map[tree_position_y-y1: tree_position_y+y2, tree_position_x-x1:tree_position_x+x2] = False

            improved = True
            while improved and (time.time() - start_time_hill_climbing < 3600 or not args.extract_trees):
                improved = False
                for tree_position_idx in range(args.number_of_trees):
                    if args.extract_trees:
                        tree_position_y, tree_position_x, tree_idx = cur_best[tree_position_idx]
                        t = extracted_trees[tree_idx]
                        y1, x1 = t.size(0) // 2, t.size(1) // 2
                        y2, x2 = t.size(0) - y1, t.size(1) - x1
                    else:
                        tree_position_y, tree_position_x = cur_best[tree_position_idx]
                    other_trees = torch.concat([cur_best[:tree_position_idx], cur_best[tree_position_idx+1:]], dim=0)
                    fitness = []
                    neighbor_trees = []
                    for y in [-1, 0, 1]:
                        for x in [-1, 0, 1]:
                            if y == 0 and x == 0:
                                continue
                            if args.extract_trees:
                                cf_tree_convoluted_valid_map = tree_convoluted_valid_map.clone()
                                cf_tree_convoluted_valid_map[tree_position_y-y1: tree_position_y+y2, tree_position_x-x1:tree_position_x+x2] = True
                                if tree_position_y+y-y1 < 0 or tree_position_y+y+y2 > cf_tree_convoluted_valid_map.size(0) or tree_position_x+x-x1<0 or tree_position_x+x+x2 > cf_tree_convoluted_valid_map.size(0):
                                    continue
                                if not torch.all(cf_tree_convoluted_valid_map[tree_position_y + y-y1: tree_position_y + y+y2, tree_position_x + x-x1:tree_position_x + x+x2]):
                                    continue
                                neighbor_tree = torch.Tensor([tree_position_y+y,tree_position_x+x,tree_idx]).int()
                            else:
                                if not (0 <= tree_position_y+y<convoluted_valid_map.size(0)):
                                    continue
                                if not (0 <= tree_position_x+x<convoluted_valid_map.size(1)):
                                    continue
                                if not convoluted_valid_map[tree_position_y+y,tree_position_x+x]:
                                    continue
                                neighbor_tree = torch.Tensor([tree_position_y+y,tree_position_x+x]).int()
                                if not idx_not_in_list(other_trees, neighbor_tree, args.crown_diameter):
                                    continue
                            neighbor_trees.append(neighbor_tree)
                            fitness.append(fitness_func_fn(torch.cat([neighbor_tree.unsqueeze(0), other_trees], dim=0).view(-1).numpy(), None)*-1)
                    if len(fitness) > 0 and np.amin(fitness) < cur_best_fitness:
                        improved = True
                        cur_best_fitness = np.amin(fitness)
                        if args.extract_trees: # free up space for tree
                            tree_position_y, tree_position_x, tree_idx = cur_best[tree_position_idx]
                            t = extracted_trees[tree_idx]
                            y1, x1 = t.size(0) // 2, t.size(1) // 2
                            y2, x2 = t.size(0) - y1, t.size(1) - x1
                            tree_convoluted_valid_map[tree_position_y-y1: tree_position_y+y2, tree_position_x-x1:tree_position_x+x2] = True
                        cur_best[tree_position_idx] = neighbor_trees[np.argmin(fitness)]
                        if args.extract_trees: # place tree and block it
                            tree_position_y, tree_position_x, tree_idx = cur_best[tree_position_idx]
                            tree_convoluted_valid_map[tree_position_y-y1: tree_position_y+y2, tree_position_x-x1:tree_position_x+x2] = False

            if cur_best_fitness < cur_best_fitness_prev:
                if "genetic" in args.method:
                    if args.extract_trees:
                        cur_bests[-1] = cur_best.reshape(-1, 3).int()
                    else:
                        cur_bests[-1] = cur_best.reshape(-1, 2).int()
                    cur_best_fitness_list[-1] = fitness_func_fn(cur_bests[-1].clone().view(-1).numpy(), None)*-1
                else:
                    if args.extract_trees:
                        cur_bests.append(cur_best.reshape(-1, 3).int())
                    else:
                        cur_bests.append(cur_best.reshape(-1, 2).int())
                    cur_best_fitness_list.append(fitness_func_fn(cur_bests[-1].clone().view(-1).numpy(), None)*-1)
        
        while len(cur_bests) > 5: # only keep the best five
            indices = np.argsort(cur_best_fitness_list)[:5]
            cur_bests = [cur_bests[idx] for idx in indices]
            cur_best_fitness_list = [cur_best_fitness_list[idx] for idx in indices]

        torch.save(
            cur_bests[np.argmin(cur_best_fitness_list)] 
            if args.extract_trees 
            else cur_bests[np.argmin(cur_best_fitness_list)] + args.crown_diameter // 2, 
            args.output_path / f"{args.method}_trees_inter.pth"
        )
    
    final_unraveled_indices = cur_bests[np.argmin(cur_best_fitness_list)]
    if not args.extract_trees:
        final_unraveled_indices += args.crown_diameter // 2

# plant trees
tree = compute_tree(args.crown_diameter, args.tree_height).to(device)
spatial_meta = plant_trees(final_unraveled_indices, spatial_meta, eval_model, extracted_trees if args.extract_trees else tree, lcc=args.with_lcc).detach()
end = time.time()
LOG_DICT["optimization_time"] = end-start
torch.save(final_unraveled_indices, args.output_path / f"{args.method}_trees.pth")

# evaluation
start = time.time()
with torch.no_grad():
    save_outputs = [] if input_temporal_t.size(0) < 10000 or args.save_timesteps else torch.zeros((1,) + spatial_meta.shape[1:])
    for outer in trange(0, input_temporal_t.size(0), args.batch_size, leave=False):
        outputs = forward(eval_model, spatial_meta, input_temporal_t[outer:outer+args.batch_size], statistics=utilities.STATISTICS["Tmrt"])
        if input_temporal_t.size(0) < 10000 or args.save_timesteps:
            save_outputs.append(outputs.detach().cpu())
        else:
            save_outputs += outputs.detach().cpu().squeeze().sum(dim=0).unsqueeze(0)
if input_temporal_t.size(0) < 10000 or args.save_timesteps:
    after_all_outputs = torch.concat(save_outputs, dim=0).squeeze()
else:
    after_all_outputs = save_outputs / input_temporal_t.size(0)
end = time.time()
LOG_DICT["evaluation_time"] = end-start

torch.save(after_all_outputs, args.output_path / f"{args.method}_outputs.pth")

LOG_DICT["tmrt_after"] = compute_mean_tmrt(after_all_outputs, valid_map.cpu())
LOG_DICT["tmrt_mean_difference"] = prev_mean_tmrt-compute_mean_tmrt(after_all_outputs, valid_map.cpu())

veg_after = deepcopy(spatial_meta[DSM_VEG_INDEX].detach().cpu().numpy())
np.save(args.output_path / f"veg_after_{args.method}.npy", veg_after)

tmrt_after= deepcopy(after_all_outputs.mean(dim=0).detach().cpu())
np.save(args.output_path / f"tmrt_after_{args.method}.npy", tmrt_after)

if args.plot:
    plt.imshow(veg_after)
    plt.colorbar()
    plt.savefig(args.output_path / f"veg_after_{args.method}.png")
    plt.close()

    plt.imshow(veg_after - veg_before)
    plt.colorbar()
    plt.savefig(args.output_path / f"veg_diff_{args.method}.png")
    plt.close()

    img = np.ma.masked_where(~valid_map.cpu().numpy(), tmrt_before)
    plt.imshow(img)
    plt.colorbar()
    plt.savefig(args.output_path / f"tmrt_after_{args.method}.png")
    plt.close()

    img = np.ma.masked_where(~valid_map.cpu().numpy(), tmrt_before - tmrt_after)
    plt.imshow(img)
    plt.colorbar()
    plt.savefig(args.output_path / f"tmrt_diff_{args.method}.png")
    plt.close()

print(LOG_DICT)
with open(args.output_path / f"{args.method}.json", "w", encoding="utf-8") as f:
    json.dump(LOG_DICT, f, indent=4)
