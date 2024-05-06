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

lib contains code of partial GW solvers, GW-based methods for shape matching and pu-learning.
lib/unbalanced_gromov is imported from [Unbalanced-GW](https://github.com/thibsej/unbalanced_gromov_wasserstein) and the lib/primal_pgw is imported from [primal-partial-GW](https://github.com/lchapel/partial-GW-for-PU).
