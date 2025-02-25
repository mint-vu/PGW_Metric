#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:58:27 2020

"""

import time
import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist

import ot
from ot.gromov import gromov_wasserstein
from . import utils


def compute_cost_matrices(P, U, prior, nb_dummies=0):
    """Compute the cost matrices (C = C(x_i, y_j), Cs = C^s(x_i, x_k),
    Ct = C^t(y_j, y_l)) and the weigths (uniform weights only).

    Parameters
    ----------
    P: pandas dataframe, shape=(n_p, d_p)
        Positive dataset

    U: pandas dataframe, shape=(n_u, d_u)
        Unlabeled dataset

    prior: percentage of positives on the dataset (s)

    nb_dummies: number of dummy points, default: no dummies
        (to avoid numerical instabilities of POT)

    Returns
    -------
    numpy.array of shape (n_p, n_u) if d_p=d_u, None otherwise
        inter-domain matrix

    numpy.array of shape (n_p, n_p)
        intra-cost matrix for P dataset

    numpy.array of shape (n_u, n_u)
        intra-cost matrix for U dataset

    numpy.array of len (n_d+n_dummies)
        weigths of the P dataset + dummies

    numpy.array of len (n_u)
        weigths of the U dataset
    """
    # Positive dataset with dummy points
    n_unl_pos = int(U.shape[0] * prior)
    P_ = P.copy()
    P_ = np.vstack([P_, np.zeros((nb_dummies, P.shape[1]))])

    # weigths
    mu = (np.ones(len(P_)) / (len(P_) - nb_dummies)) * (n_unl_pos / len(U))
    if nb_dummies > 0:
        mu[-nb_dummies:] = (1 - np.sum(mu[:-nb_dummies])) / nb_dummies
    else:
        mu = mu / np.sum(mu)
    nu = np.ones(len(U)) / len(U)

    # intra-domain
    C1 = sp.spatial.distance.cdist(P_, P_)
    C2 = sp.spatial.distance.cdist(U, U)
    if nb_dummies > 0:
        C1[:, -nb_dummies:] = C1[-nb_dummies:, :] = C2.max() * 1e2
        C1[-nb_dummies:, -nb_dummies:] = 0

    # inter-domain
    if P_.shape[1] == U.shape[1]:
        C = sp.spatial.distance.cdist(P_, U)
        if nb_dummies > 0:
            C[-nb_dummies:, :] = 1e2 * C[:-nb_dummies, :].max()
    else:
        C = None
    return C, C1, C2, mu, nu


def pu_w_emd(p, q, C, nb_dummies=1):
    """Compute the transport matrix that solves an EMD in the context of
    partial-Wasserstein (with dummy points) + pu learning (with group
    constraints)

    Parameters
    ----------
    p: array of len (n_p+nb_dummies)
        weights of the source (positives)

    q: array of len (n_u)
        weights of the source (unlabeled)

    nb_dummies: number of dummy points, default: 1
        (to avoid numerical instabilities of POT)

    Returns
    -------
    numpy.array of shape (n_p+nb_dummies, n_u)
        transport matrix
    """
    lstlab = np.array([0, 1])
    labels_a = np.append(
        np.array([0] * (len(p) - nb_dummies)), np.array([1] * (nb_dummies))
    )

    def f(G):
        res = 0
        for i in range(G.shape[1]):
            for lab in lstlab:
                temp = G[labels_a == lab, i]
                res += (np.linalg.norm(temp, 1)) ** 0.5
        return res

    def df(G):
        W = np.zeros(G.shape)
        for i in range(G.shape[1]):
            for lab in lstlab:
                temp = G[labels_a == lab, i]
                W[labels_a == lab, i] = 0.5 * (np.linalg.norm(temp, 1)) ** (-0.5)
        return W

    Gc = ot.optim.cg(p, q, C, 1e6, f, df, numItermax=20)
    return Gc


def gwgrad(C1, C2, T):
    """Compute the GW gradient. Note: we can not use the trick of Peyré as
    the marginals may not sum to 1.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    numpy.array of shape (n_p+nb_dummies, n_u)
        gradient
    """
    constC1 = np.dot(C1**2 / 2, np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1)))
    constC2 = np.dot(np.dot(np.ones(C1.shape[0]).reshape(1, -1), T), C2**2 / 2)
    constC = constC1 + constC2
    A = -np.dot(C1, T).dot(C2.T)
    tens = constC + A
    return tens * 2


def gwloss(C1, C2, T):
    """Compute the GW loss.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    loss
    """
    g = gwgrad(C1, C2, T) * 0.5
    return np.sum(g * T)


def pu_gw_emd(
    C1, C2, p, q, nb_dummies=1, group_constraints=False, G0=None, log=False, max_iter=20
):
    """Compute the transport matrix that solves an EMD in the context of
    partial-Gromov Wasserstein (with dummy points) + pu learning (with group
    constraints or not)

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    p: array of len (n_p+nb_dummies)
        weights of the source (unlabeled)

    q: array of len (n_u)
        weights of the target (unlabeled)

    nb_dummies: number of dummy points, default: 1
        (to avoid numerical instabilities of POT)

    group_constraints: either we want to enforce groups (default: False)

    G0 : array of shape(n_p+nb_dummies, n_u) (default: None)
        Initialisation of the transport matrix

    log: wether we return the loss of each iteration (default: False)

    max_iter: maximum number of iterations (default: 20)

    Returns
    -------
    numpy.array of shape (n_p+nb_dummies, n_u)
        transport matrix

    numpy.array of the history of the loss along the iterations
    """

    # initialisation of the transport matrix
    if G0 is None:
        G0 = np.outer(p, q)
        G0[-nb_dummies:] = 0
    else:
        mask = G0 < 1e-7
        G0[mask] = 0
        G0[-nb_dummies:] = 0

    loop = True
    it = 0
    t_loss = []

    # step 8 of alg. 1
    C1_grad = C1.copy()
    C1_grad[-nb_dummies:, -nb_dummies:] = np.quantile(
        C1[:-nb_dummies, :-nb_dummies], 0.75
    )

    C1_nodummies = C1[:-nb_dummies, :-nb_dummies]
    C2_nodummies = C2.copy()

    loss = gwloss(
        C1_nodummies, C2_nodummies, G0[: C1_nodummies.shape[0], : C2_nodummies.shape[0]]
    )

    while loop:
        # Collect the appropriate submatrix
        idx_pos = np.where(np.sum(G0, axis=0) > 1e-7)[0]
        mask = G0 > 1e-7

        C2_current = C2[idx_pos, :]
        C2_current = C2_current[:, idx_pos]

        it += 1
        # Compute the gradient of the active set (step 7 of alg.1)
        G0 = G0[:, idx_pos]
        M = gwgrad(C1_nodummies, C2_current, G0[:-nb_dummies])

        # add the dummy poit + negative points (step 8 of alg.1)
        M_emd = np.ones((len(p), len(q))) * np.quantile(M[M > 1e-7], 0.75)

        # step 9 of alg.1
        idx = 0
        for i in idx_pos:
            M_emd[:-nb_dummies, i] = M[:, idx]
            idx += 1
        M_emd[-nb_dummies:] = M_emd.max() * 1e3
        M_emd = np.asarray(M_emd, dtype=np.float64)

        # step 10 of alg.1
        if group_constraints is False:
            Gc, logemd = ot.lp.emd(p, q, M_emd, log=True)
        else:
            Gc = pu_w_emd(p, q, M_emd, nb_dummies)
            logemd = {}
            logemd["warning"] = None

        if logemd["warning"] is not None:
            loop = False
            print("Error in the EMD!!!!!!!")

        G0 = Gc

        prevloss = loss
        loss = gwloss(
            C1_nodummies,
            C2_nodummies,
            G0[: C1_nodummies.shape[0], : C2_nodummies.shape[0]],
        )
        t_loss.append(loss)
        if it > max_iter:
            loop = False

        if (it > 2) and (np.abs(prevloss - loss) < 10e-15):
            loop = False

    if log:
        return G0, t_loss
    else:
        return G0


def compute_perf_emd(dataset_p, dataset_u, n_pos, n_unl, prior, nb_reps, nb_dummies=1):
    """Compute the performances of running the partial-W for a PU learning
    task on a given dataset several times

    Parameters
    ----------
    dataset_p: name of the dataset among which the positives are drawn

    dataset_u: name of the dataset among which the unlabeled are drawn

    n_pos: number of points in the positive dataset

    n_unl: number of points in the unlabeled dataset

    prior: percentage of positives on the dataset (s)

    nb_resp: number of runs

    nb_dummies: number of dummy points, default: no dummies
        (to avoid numerical instabilities of POT)

    Returns
    -------
    dict with:
        - the class prior
        - the performances of the p-w (avg among the repetitions)
        - the performances of the p-w with group constraints (avg)
        - the list of all the nb_reps performances of the p-w
        - the list of all the nb_reps performances of the p-w with groups
    """
    perfs = {}
    perfs["class_prior"] = prior
    perfs["emd"] = 0
    perfs["emd_groups"] = 0
    perfs_list = {}
    perfs_list["emd"] = []
    perfs_list["emd_groups"] = []
    start_time = time.time()
    for i in range(nb_reps):
        P, U, y_u = utils.draw_p_u_dataset_scar(
            dataset_p, dataset_u, n_pos, n_unl, prior, i
        )  # seed=i
        Ctot, _, _, mu, nu = compute_cost_matrices(P, U, prior, nb_dummies)
        nb_unl_pos = int(np.sum(y_u))

        transp_emd = ot.emd(mu, nu, Ctot)
        y_hat = np.ones(len(y_u))
        sum_dummies = np.sum(transp_emd[-nb_dummies:], axis=0)
        y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0
        perfs_list["emd"].append(np.mean(y_u == y_hat))
        perfs["emd"] += np.mean(y_u == y_hat)

        transp_emd_group = pu_w_emd(mu, nu, Ctot, nb_dummies)
        y_hat = np.ones(len(y_u))
        sum_dummies = np.sum(transp_emd_group[-nb_dummies:], axis=0)
        y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0
        perfs_list["emd_groups"].append(np.mean(y_u == y_hat))
        perfs["emd_groups"] += np.mean(y_u == y_hat)

    perfs["emd"] = perfs["emd"] / nb_reps
    perfs["emd_groups"] = perfs["emd_groups"] / nb_reps
    perfs["time"] = time.time() - start_time
    return perfs, perfs_list


def compute_perf_pgw(dataset_p, dataset_u, n_pos, n_unl, prior, nb_reps, nb_dummies=1):
    """Compute the performances of running the partial-GW for a PU learning
    task on a given dataset several times

    Parameters
    ----------
    dataset_p: name of the dataset among which the positives are drawn

    dataset_u: name of the dataset among which the unlabeled are drawn

    n_pos: number of points in the positive dataset

    n_unl: number of points in the unlabeled dataset

    prior: percentage of positives on the dataset (s)

    nb_resp: number of runs

    nb_dummies: number of dummy points, default: no dummies
        (to avoid numerical instabilities of POT)

    Returns
    -------
    dict with:
        - the class prior
        - the performances of the p-gw (avg among the repetitions)
        - the performances of the p-gw with group constraints (avg)
        - the list of all the nb_reps performances of the p-gw
        - the list of all the nb_reps performances of the p-gw with groups
    """

    perfs = {}
    perfs["class_prior"] = prior
    perfs["pgw"] = 0
    perfs["pgw_groups"] = 0
    perfs_list = {}
    perfs_list["pgw"] = []
    perfs_list["pgw_groups"] = []
    start_time = time.time()
    for i in range(nb_reps):
        P, U, y_u = utils.draw_p_u_dataset_scar(
            dataset_p, dataset_u, n_pos, n_unl, prior, i
        )  #     seed=i
        Ctot, C1, C2, mu, nu = compute_cost_matrices(P, U, prior, nb_dummies)
        nb_unl_pos = int(np.sum(y_u))

        Ginit = []
        if Ctot is not None:
            Ginit.append(ot.emd(mu, nu, Ctot))  # We can init. with the EMD
        Ginit.append(None)
        _, Cs, _, ps, pt = compute_cost_matrices(P, U, prior, 0)
        Ginit = Ginit + initialisation_gw(ps, pt, Cs, U, prior, 10, nb_dummies)

        best_loss = 0
        # We test several init (emd if possible, outer product, barycenter)
        # and keep the one that provides the best loss
        for init in Ginit:
            transp_emd, t_loss = pu_gw_emd(
                C1, C2, mu, nu, nb_dummies, group_constraints=False, G0=init, log=True
            )
            y_hat = np.ones(len(y_u))
            sum_dummies = np.sum(transp_emd[-nb_dummies:], axis=0)
            y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0
            t_loss[-1] = np.mean(y_u == y_hat)
            if t_loss[-1] > best_loss:
                best_loss = t_loss[-1]
                transp_emd_best = transp_emd.copy()

        y_hat = np.ones(len(y_u))
        sum_dummies = np.sum(transp_emd_best[-nb_dummies:], axis=0)
        y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0
        perfs_list["pgw"].append(np.mean(y_u == y_hat))
        perfs["pgw"] += np.mean(y_u == y_hat)

        transp_emd_group = pu_gw_emd(
            C1, C2, mu, nu, nb_dummies, group_constraints=True, G0=None
        )
        y_hat = np.ones(len(y_u))
        sum_dummies = np.sum(transp_emd_group[-nb_dummies:], axis=0)
        y_hat[np.argsort(sum_dummies)[nb_unl_pos:]] = 0
        perfs_list["pgw_groups"].append(np.mean(y_u == y_hat))
        perfs["pgw_groups"] += np.mean(y_u == y_hat)

    perfs["pgw"] = perfs["pgw"] / nb_reps
    perfs["pgw_groups"] = perfs["pgw_groups"] / nb_reps
    perfs["time"] = time.time() - start_time
    return perfs, perfs_list


def initialisation_gw(p, q, Cs, U, prior=0, nb_init=5, nb_dummies=1):
    """Initialisation of GW. From a barycenter, it gives a first shot for
    the transport matrix

    Parameters
    ----------
    p: array of len (n_p)
        weights of the source (positives !! with no dummies !!)

    q: array of len (n_u)
        weights of the target (unlabeled)

    Cs: array of shape (n_p,n_p)
        intra-source (P) cost matrix (!! no dummies !!)

    U: array of shape (n_u,d_u)
        U dataset

    prior: percentage of positives on the dataset (s)

    n_init: number of atoms on the barycenter

    nb_dummies: number of dummy points, default: 1
        (to avoid numerical instabilities of POT)

    Returns
    -------
    list of numpy.array of shape (n_p+, n_u)
        list of potentialtransport matrix initialisation

    """
    if nb_init > 2:
        res, _, _ = wass_bary_coarsening(
            nb_init, np.array(U), pt=np.ones(U.shape[0]) / (U.shape[0])
        )
    else:
        res, _, _ = wass_bary_coarsening(
            nb_init,
            np.array(U),
            pt=np.ones(U.shape[0]) / (U.shape[0]),
            pb=[prior, 1 - prior],
        )
    idx = []
    l_gamma = []
    for i in range(nb_init):
        idx = np.where(res[i, :] > 1e-5)[0]
        gamma = np.zeros((len(p) + nb_dummies, len(q)))

        Ct_0 = cdist(U.iloc[idx], U.iloc[idx])
        gamma1 = gromov_wasserstein(
            Cs, Ct_0, p, np.ones(Ct_0.shape[0]) / Ct_0.shape[0], "square_loss"
        )
        gamma1 /= np.sum(gamma1)
        for i in range(len(idx)):
            gamma[:-nb_dummies, idx[i]] = gamma1[:, i]
        l_gamma.append(gamma)
    return l_gamma


def wass_bary_coarsening(n, U, pt=None, pb=None):
    """Computation of a GW barycenter

    Parameters
    ----------
    n: number of atoms of the barycenter

    U: array of shape (n_u,d_u)
        U dataset

    pt: array of len (n_u) (default 1/n_u)
        weights of the target (unlabeled)

    pb: array of len (n) (default 1/n)
        weights of the barycenter

    Returns
    -------
    numpy.array of shape (n, n_u)
        transport matrix between the barycenter and U

    numpy.array of shape (n, n)
        features of the barycenter

    numpy.array of shape (1,1)
        gw cost
    """
    if pb is None:
        pb = np.ones(n) / n
    if pt is None:
        pt = np.ones(U.shape[0]) / U.shape[0]

    U = np.asarray(U)
    pb_diag = np.diag(pb)
    T = np.outer(pb, pt)

    cpt = 0
    err = 1
    tol = 1e-9
    max_iter = 100

    while (err > tol) and cpt < max_iter:
        Tprev = T.copy()
        X = np.dot(U.T, T.T).dot(pb_diag)
        M = sp.spatial.distance.cdist(X.T, U)
        if cpt == 0:
            Mmax = M.max()
        T, logemd = ot.emd(pb, pt / np.sum(pt), M / Mmax, log=True)
        if logemd["warning"] is not None:
            print(logemd["warning"])
        err = np.linalg.norm(T - Tprev)
        cpt += 1
    return T, X.T, np.sum(np.sum(M * T))
