# Partial Gromov-Wasserstein Metric

This repository contains the accompanying code/experiments for the paper [Partial Gromov-Wasserstein Metric](https://arxiv.org/abs/2402.03664).


## Abstract

The Gromov-Wasserstein (GW) distance has gained increasing interest in the machine learning community in recent years, as it allows for the comparison of measures in different metric spaces. To overcome the limitations imposed by the equal mass requirements of the classical GW problem, researchers have begun exploring its application in unbalanced settings. However, Unbalanced GW (UGW) can only be regarded as a discrepancy rather than a rigorous metric/distance between two metric measure spaces (mm-spaces). In this paper, we propose a particular case of the UGW problem, termed Partial Gromov-Wasserstein (PGW). We establish that PGW is a well-defined metric between mm-spaces and discuss its theoretical properties, including the existence of a minimizer for the PGW problem and the relationship between PGW and GW, among others. We then propose two variants of the Frank-Wolfe algorithm for solving the PGW problem and show that they are mathematically and computationally equivalent. Moreover, based on our PGW metric, we introduce the analogous concept of barycenters for mm-spaces. Finally, we validate the effectiveness of our PGW metric and related solvers in applications such as shape matching, shape retrieval, and shape interpolation, comparing them against existing baselines.

## Required Packages

We suggest installing [Pytorch](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), [PythonOT](https://pythonot.github.io/), [scipy](https://scipy.org/),
[numba](https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html), and [scikit-learn](https://scikit-learn.org/stable/) to reproduce the experiments in this repository.

Alternatively, you can create the necessary Conda environment with the following command:
```bash
conda env create -f environment.yml
```

## Repository Structure

`lib/` contains code of partial GW solvers and GW-based methods for all experiments. See `lib/README.md` for references.

### Numerical experiments
- Run `shape_matching/shape_matching.ipynb` to see the numerical result of shape matching.
- Run `interpolation/barycenter.ipynb` to see the numerical result of point cloud interpolation. 
    - Run `interpolation/barycenter_visual.ipynb` to see a visulization. 
- Run `run_time/run_time.ipynb` to see the wall-clock time comparison. 
- Run `shape_retrieval/bone_star_exp.ipynb` and `shape_retrieval/synthetic_data_exp.ipynb` to see the results of shape retrieval. 


## Citation

```
@inproceedings{bai2024partialgromovwassersteinmetric,
      title={Partial Gromov-Wasserstein Metric}, 
      author={Yikun Bai and Rocio Diaz Martin and Abihith Kothapalli and Hengrong Du and Xinran Liu and Soheil Kolouri},
      year={2024},
}
```