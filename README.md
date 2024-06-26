# Installation:

Partial Gromov Wasserstein

Example code for partial Gromov Wasserstein solver. The code reproduces the all the numerical examples in the paper.

# Required package:

We suggest to install [Pytorch](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html), [PythonOT](https://pythonot.github.io/), [scipy](https://scipy.org/),
[numba](https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html), [sk-learning](https://scikit-learn.org/stable/).

Can create necessary Conda environment with 
```bash
conda env create -f environment.yml
```

# Outline of repository

`lib/` contains code of partial GW solvers and GW-based methods for all experiments. See `lib/README.md` for references.

# Numerical experiments
- Run `shape_matching.ipynb` to see the numerical result of shape matching.
- Run `barycenter.ipynb` to see the numerical result of point cloud interpolation. 
    - Run `barycenter_visual.ipynb` to see the visulization. 
- Run `run_time.ipynb` to see the wall-clock time comparison. 
- Run `pu_learning` to see the numerical result in PU learning experiment. 
- Run `shape_retrieval/bone_star_exp.ipynb` and `shape_retrieval/synthetic_data_exp.ipynb` to see the results of shape retrieval. 


