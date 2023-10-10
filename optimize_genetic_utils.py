from __future__ import annotations

import random
import numpy as np
import torch

from optimize_utils import idx_not_in_list, plant_trees, forward_without_svf, compute_mean_tmrt

MUTATIONS_PER_STEP = 2

def create_initial_population(size: int, crown_diameter: int, number_of_trees: int, valid_map: torch.Tensor, p: torch.Tensor, all_possible_tree_locations: list, trees=None):
    initial_population = []
    for _ in range(size):
        gene = []
        if trees is not None:
            for tree_idx, t in enumerate(trees):
                y1, x1 = t.size(0) // 2, t.size(1) // 2
                y2, x2 = t.size(0) - y1, t.size(1) - x1
                new_tree_valid_map = valid_map.clone()
                if p is not None:
                    indices = np.random.choice(np.arange(len(all_possible_tree_locations)), size=200, replace=False, p=p)
                else:
                    indices = np.random.choice(np.arange(len(all_possible_tree_locations)), size=200, replace=False)
                for idx in indices:
                    y, x = all_possible_tree_locations[idx]
                    if y-y1 < 0 or y+y2 > new_tree_valid_map.size(0) or x-x1<0 or x+x2 > new_tree_valid_map.size(0):
                        continue
                    if torch.all(new_tree_valid_map[y-y1: y+y2, x-x1:x+x2]):
                        new_tree_valid_map[y-y1: y+y2, x-x1:x+x2] = False
                        gene.append(torch.Tensor([y, x, tree_idx]).int())
                        break
        else:
            while len(gene) < number_of_trees:
                if p is not None:
                    indices = np.random.choice(np.arange(len(all_possible_tree_locations)), size=200, replace=False, p=p)
                else:
                    indices = np.random.choice(np.arange(len(all_possible_tree_locations)), size=200, replace=False)
                for idx in indices:
                    idx_pair = torch.Tensor(all_possible_tree_locations[idx]).int()
                    if valid_map[idx_pair[0], idx_pair[1]] and idx_not_in_list(gene, idx_pair, crown_diameter):
                        gene.append(idx_pair)
                    if len(gene) == number_of_trees:
                        break
        # transform gene since pygad expects 2d inputs...
        initial_population.append(torch.stack(gene, dim=0).view(-1).numpy())
    return initial_population

def fitness_func(model, tree, spatial_meta, valid_map, batch_size, crown_diameter, temporal_meta=None, statistics=None, lcc: bool = True) -> np.ndarray:  # pylint: disable=unused-argument
    model = model
    tree = tree
    spatial_meta = spatial_meta
    temporal_meta = temporal_meta
    valid_map = valid_map.cpu()
    batch_size = batch_size
    crown_diameter = crown_diameter
    statistics=statistics
    lcc = lcc
    def inner_func(solution, solution_idx):
        nonlocal model
        nonlocal tree
        nonlocal spatial_meta
        nonlocal temporal_meta
        nonlocal valid_map
        nonlocal batch_size
        nonlocal crown_diameter
        nonlocal statistics
        nonlocal lcc
        if isinstance(tree, list):
            unraveled_indices = solution.reshape(-1, 3).astype(np.int32)
        else:
            unraveled_indices = solution.reshape(-1, 2).astype(np.int32) + crown_diameter // 2
        with torch.no_grad():
            new_spatial_meta = torch.clone(spatial_meta)
            new_spatial_meta = plant_trees(unraveled_indices, new_spatial_meta, model, tree, lcc=lcc).detach()
            if temporal_meta is None:
                new_output = forward_without_svf(model, new_spatial_meta, statistics=statistics).squeeze(dim=1)
            else:
                save_outputs = []
                for outer in range(0, temporal_meta.size(0), batch_size):
                    outputs = forward_without_svf(model, new_spatial_meta, temporal_meta[outer:outer+batch_size], statistics)
                    save_outputs.append(outputs.detach().cpu())
                new_output = torch.concat(save_outputs, dim=0).squeeze().cpu().detach()
            fitness = compute_mean_tmrt(new_output, valid_map)
        return fitness * -1  # pygad is doing maximization internally
    return inner_func


def crossover_func(
    crown_diameter: int, number_of_trees: int, valid_map: torch.Tensor, p: torch.Tensor, all_possible_tree_locations: list, trees: list = None
) -> callable:  # pylint: disable=unused-argument
    crown_diameter = crown_diameter
    number_of_trees = number_of_trees
    valid_map = valid_map
    p = p
    all_possible_tree_locations = all_possible_tree_locations
    trees = trees
    def inner_func(parents, offspring_size, ga_instance):
        # single-point crossover function inspired by https://pygad.readthedocs.io/en/latest/pygad.html#user-defined-crossover-mutation-and-parent-selection-operators
        nonlocal crown_diameter
        nonlocal number_of_trees
        nonlocal valid_map
        nonlocal p
        nonlocal all_possible_tree_locations
        nonlocal trees

        offspring = []
        idx = 0
        while len(offspring) < offspring_size[0]:
            parent1 = parents[idx % parents.shape[0], :].copy().astype(np.int32)
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy().astype(np.int32)
            if trees is not None:
                split_points = list(range(offspring_size[1] // 3))
            else:
                split_points = list(range(offspring_size[1] // 2))
            random.shuffle(split_points)
            for random_split_point in split_points:
                valid_crossover = True
                if trees is not None:
                    new_tree_valid_map = valid_map.clone()
                    for y, x, t_idx in parent1[:random_split_point * 3].reshape(
                        parent1[:random_split_point * 3].shape[0] // 3, 3
                    ):
                        y1, x1 = trees[t_idx].size(0) // 2, trees[t_idx].size(1) // 2
                        y2, x2 = trees[t_idx].size(0) - y1, trees[t_idx].size(1) - x1
                        new_tree_valid_map[y-y1: y+y2, x-x1:x+x2] = False
                    for y, x, t_idx in parent2[random_split_point * 3 :].reshape(
                        parent2[random_split_point * 3 :].shape[0] // 3, 3
                    ):
                        y1, x1 = trees[t_idx].size(0) // 2, trees[t_idx].size(1) // 2
                        y2, x2 = trees[t_idx].size(0) - y1, trees[t_idx].size(1) - x1
                        if not torch.all(new_tree_valid_map[y-y1: y+y2, x-x1:x+x2]):
                            valid_crossover = False
                            break
                        new_tree_valid_map[y-y1: y+y2, x-x1:x+x2] = False
                else:
                    for y, x in parent2[random_split_point * 2 :].reshape(
                        parent2[random_split_point * 2 :].shape[0] // 2, 2
                    ):
                        if not (valid_map[y, x].item() and idx_not_in_list(full_list=torch.from_numpy(parent1[:random_split_point * 2].reshape(-1, 2)), potential_new_pair=torch.Tensor([y, x]).int(), crown_diameter=crown_diameter)):
                            valid_crossover = False
                            break
                if valid_crossover:
                    spring = np.empty_like(parent1)
                    if trees is not None:
                        spring[:random_split_point * 3] = parent1[:random_split_point * 3]
                        spring[random_split_point * 3 :] = parent2[random_split_point * 3 :]
                    else:
                        spring[:random_split_point * 2] = parent1[:random_split_point * 2]
                        spring[random_split_point * 2 :] = parent2[random_split_point * 2 :]
                    offspring.append(spring)
                    idx += 1
                    break
                else:
                    offspring += create_initial_population(
                        size=1,
                        crown_diameter=crown_diameter,
                        number_of_trees=number_of_trees,
                        valid_map=valid_map,
                        p=p,
                        all_possible_tree_locations=all_possible_tree_locations,
                        trees=trees,
                    )
                if len(offspring) == offspring_size[0]:
                    break
        return np.array(offspring)
    return inner_func


def mutation_func(crown_diameter: int, valid_map: torch.Tensor, trees: list = None):  # pylint: disable=unused-argument
    crown_diameter = crown_diameter
    valid_map = valid_map
    trees = trees
    
    def inner_func(offspring, ga_instance):
        # random mutation function inspired by https://pygad.readthedocs.io/en/latest/pygad.html#user-defined-crossover-mutation-and-parent-selection-operators

        nonlocal crown_diameter
        nonlocal valid_map

        for chromosome_idx in range(offspring.shape[0]):
            if trees is not None:
                new_tree_valid_map = valid_map.clone()
                for y, x, tree_idx in offspring[chromosome_idx].reshape(-1, 3):
                    y1, x1 = trees[tree_idx].size(0) // 2, trees[tree_idx].size(1) // 2
                    y2, x2 = trees[tree_idx].size(0) - y1, trees[tree_idx].size(1) - x1
                    new_tree_valid_map[y-y1: y+y2, x-x1:x+x2] = False

            p = MUTATIONS_PER_STEP / (2*len(trees)) if trees is not None else MUTATIONS_PER_STEP / offspring.shape[1]
            for gene_idx in range(2*len(trees) if trees is not None else offspring.shape[1]):
                if random.random() < p:
                    motion = random.choice(range(-crown_diameter, crown_diameter+1))
                    if trees is not None:
                        tree_idx = gene_idx // 2
                        y1, x1 = trees[tree_idx].size(0) // 2, trees[tree_idx].size(1) // 2
                        y2, x2 = trees[tree_idx].size(0) - y1, trees[tree_idx].size(1) - x1
                        true_gene_idx = (gene_idx // 2)*3 + (gene_idx%2)
                        if gene_idx % 2 == 0:
                            y_before = offspring[chromosome_idx][true_gene_idx]
                            x_before = offspring[chromosome_idx][true_gene_idx+1]
                        else:
                            y_before = offspring[chromosome_idx][true_gene_idx-1]
                            x_before = offspring[chromosome_idx][true_gene_idx]
                        new_tree_valid_map[y_before-y1: y_before+y2, x_before-x1:x_before+x2] = True
                        if gene_idx % 2 == 0:
                            y = y_before + motion
                            x = x_before
                        else:
                            y = y_before
                            x = x_before + motion
                        if y-y1 < 0 or y+y2 > new_tree_valid_map.size(0) or x-x1<0 or x+x2 > new_tree_valid_map.size(0):
                            continue
                        if torch.all(new_tree_valid_map[y-y1: y+y2, x-x1:x+x2]):
                            new_tree_valid_map[y-y1: y+y2, x-x1:x+x2] = False
                            if gene_idx % 2 == 0:
                                offspring[chromosome_idx][true_gene_idx] = y
                            else:
                                offspring[chromosome_idx][true_gene_idx] = x
                        else:
                            new_tree_valid_map[y_before-y1: y_before+y2, x_before-x1:x_before+x2] = False
                    else:
                        if gene_idx % 2 == 0:
                            new_x = np.clip(
                                offspring[chromosome_idx][gene_idx] + motion,
                                0,
                                valid_map.shape[-2]-1,
                            )
                            others = np.concatenate((offspring[chromosome_idx, :gene_idx], offspring[chromosome_idx, gene_idx+2:])).reshape(-1, 2)
                            if valid_map[new_x, offspring[chromosome_idx][gene_idx+1]].item() and idx_not_in_list(full_list=torch.from_numpy(others), potential_new_pair=torch.Tensor([new_x, offspring[chromosome_idx][gene_idx+1]]).int(), crown_diameter=crown_diameter):
                                offspring[chromosome_idx][gene_idx] = new_x
                        else:
                            new_y = np.clip(
                                offspring[chromosome_idx][gene_idx] + motion,
                                0,
                                valid_map.shape[-1]-1,
                            )
                            others = np.concatenate((offspring[chromosome_idx, :gene_idx-1], offspring[chromosome_idx, gene_idx+1:])).reshape(-1, 2)
                            if valid_map[offspring[chromosome_idx][gene_idx-1], new_y].item() and idx_not_in_list(full_list=torch.from_numpy(others), potential_new_pair=torch.Tensor([offspring[chromosome_idx][gene_idx-1], new_y]).int(), crown_diameter=crown_diameter):
                                offspring[chromosome_idx][gene_idx] = new_y
        return offspring
    return inner_func