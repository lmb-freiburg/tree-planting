# Climate-sensitive Urban Planning Through Optimization of Tree Placements

This is the code for the paper "Climate-sensitive Urban Planning Through Optimization of Tree Placements".

If this work is useful to you, please consider citing our paper:

```
@misc{schrodi2023climatesensitive,
    title={Climate-sensitive Urban Planning through Optimization of Tree Placements}, 
    author={Simon Schrodi and Ferdinand Briegel and Max Argus and Andreas Christen and Thomas Brox},
    year={2023},
    eprint={2310.05691},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Setup

To install all packages run

```bash
conda env create -n tree_planting -f environment.yml
```

and activate the environment

```bash
conda activate tree_planting
```

We are in discussions with the city of Freiburg to get permission to share the data.

## Training of models

Below we provide details for the training of the various models.

### Training of model $f_{T_{\text{mrt}}}$

This model receives spatio-temporal input to estimate point-wise $T_{\text{mrt}}$, following [Briegel et al, 2023](https://www.sciencedirect.com/science/article/pii/S2212095522002772). You can train the model by running

```bash
python train_tmrt.py \
results/tmrt_model \
--skip 1 2 3 \
--amp \
--clip_grad \
--without_aveg
```

### Training of model $f_{\text{svf}}$

This model estimates the sky view factors for vegetation given the digital surface model for vegetation. To train the model run

```bash
python train_dsmv_to_svfs.py \
results/dsmv_to_svf \
--skip 1 2 3 \
--amp \
--clip_grad \
--n_epochs 20 \
--veg_only
```

### Training of model $f_{T^{M,\phi}_{\text{mrt}}}$

Since estimating aggregated, point-wise $T_{\text{mrt}}$ is computationally very expensive, we train a model that directly estimates it.
To this end, you first need to generate the training data:

```bash
python precompute_tmrt_aggregated.py --time_period $time_period
```

Note that you have to do this only once for data generation, while during optimization we can just take the computational shortcut.
We currently support the following time periods (`$time_period`):

* hottest_day_2020
* hottest_week_2020
* year_2020
* decade_2011_2020

However, you can easily consider other time periods by filtering other periods from the pandas data frame of the ERA5 reanalysis data from 1990 till 2020.

Finally, you can train the model via

```bash
python train_tmrt_aggregated.py \
results/aggregated_tmrt_model/$time_period \
--time_period $time_period \
--skip 1 2 3 \
--amp \
--clip_grad \
--without_aveg
```

## Optimization of Tree Placements

Finally, you can optimize tree placements by running

```bash
python optimize.py \
--area $area \
--time_period $time_period \
--method $method \
--plot
```

You can optimize for the following areas (`$area`):
* 413500_5316000 (city-center)
* 414000_5318000 (new residential area)
* 409500_5316500 (medium-age residential area)
* 414000_5315000 (old residential area)
* 414000_5320000 (industrial area)

You can set the time period (`$time_period`) as described above.

Currently, we support the following methods (`$methods`):
* `random` (randomly position trees)
* `greedy` (greedy heuristic based on maximal $T_{\text{mrt}}$)
* `ils` (greedy heuristic based on maximal $\Delta T_{\text{mrt}}$)
* `genetic` (genetic algorithm)
* `ìls_genetic_hill_climbing_iterated` (ours)

## Acknowledgements

We thank the city of Freiburg for sharing spatial data (digital elevation model and digital surface models).