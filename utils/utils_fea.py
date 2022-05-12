import matplotlib

matplotlib.use('Agg')
import numpy as np
import os
# import tensorflow as tf
import sys
from time import time
# import feature_valuation.shap_utils as shap_utils
from third_party.shap_utils import insert0_2loc_i, nm_one_bits
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import jit, prange
from utils.utils import xlogx, sigmoid


def compute_mt(x, all_coa_vals, tempe):
    """
    Compute the multilinear extension at x
    """
    res = 0
    n = len(x)
    for i in range(2 ** n):

        weight = 1.0
        for k in range(n):
            if i & (1 << k):
                weight *= x[k]
            else:
                weight *= (1 - x[k])
        res += all_coa_vals[i] * weight / tempe

    return res


@jit(forceobj=True, parallel=True)
def compute_all_diffs(n, all_coalition_vals):
    # calculate all diffs
    all_diffs = np.zeros((n, 2 ** (n - 1)))

    for i in prange(n):
        for j in prange(2 ** (n - 1)):
            small_coal_id = insert0_2loc_i(i, j)
            large_coal_id = small_coal_id | (1 << i)
            all_diffs[i][j] = all_coalition_vals[int(large_coal_id)] - \
                              all_coalition_vals[int(small_coal_id)]

    # print('All diffs calculated!')
    return all_diffs


@jit(forceobj=True, parallel=True)
def compute_true_shapley(n, all_diffs):
    n_facto = np.math.factorial(n)
    # calculate true shapley
    shap_weights = np.zeros((2 ** (n - 1)))
    for j in prange(2 ** (n - 1)):
        coal_size = nm_one_bits(j)
        shap_weights[j] = np.math.factorial(coal_size) * np.math.factorial(n - coal_size - 1) / n_facto

    vals_true_shap = np.matmul(all_diffs, shap_weights)
    return vals_true_shap


@jit(nopython=True)
def compute_mt_partial_derivative(i, x, n, all_diffs):
    """
    Args:
        x:  np vector of size n
    """

    def insert0_2loc_i(i, j):
        """insert 0 to location i of j
        """
        all_ones = ~0
        left_mask = (1 << i) - 1
        left = j & left_mask
        right_mask = all_ones << i
        right = (j & right_mask) << 1
        return left | right

    ans = 0
    for j in range(2 ** (n - 1)):
        small_coal_id = insert0_2loc_i(i, j)
        weight = 1.0
        for k in range(n):
            if k == i:
                continue
            if small_coal_id & (1 << k):
                weight *= x[k]
            else:
                # if i == k:
                #     print('wow, bad i--k')
                weight *= (1 - x[k])
        ans += weight * all_diffs[i, j]
    return ans


def compute_mt_gradient(x, n, all_diffs, n_jobs=1):
    grad = Parallel(n_jobs=n_jobs, backend='threading')(delayed(compute_mt_partial_derivative)(i, x, n, all_diffs)
                                                        for i in range(n))
    return np.array(grad)


@jit(nopython=True, parallel=True)
def compute_mt_gradient_jit(x, n, all_diffs):
    grad = [compute_mt_partial_derivative(i, x, n, all_diffs) for i in prange(n)]

    return np.array(grad)


def compute_elbo(x, all_coa_vals, tempe):
    return compute_mt(x, all_coa_vals, tempe) + compute_entropy(x)


def compute_entropy(x):
    """
    Compute the entropy of q(S|x) at x
    """
    minus_x = 1 - x
    return - np.sum(xlogx(x)) - np.sum(xlogx(minus_x))


def compute_log_parti(all_coa_vals, tempe):
    return np.log(np.sum(np.exp(all_coa_vals / tempe)))


@jit(forceobj=True)
def naive_mf(init, n, all_diffs, nm_epochs=6, tempe=1):
    """
    Args:
        init: initializer
        nm_epochs
        tempe: temperature
    """
    x = init
    for _ in tqdm(range(nm_epochs)):
        for i in range(n):
            gradi = compute_mt_partial_derivative(i, x, n, all_diffs)
            x[i] = sigmoid(gradi / tempe)
    return x


def naive_fg(init, n, all_diffs, nm_epochs=6, tempe=1.0, n_jobs=1):
    """
    Args:
       init: initializer
        nm_epochs
        tempe: temperature
    """
    x = init
    for _ in tqdm(range(nm_epochs), desc="Epoch"):
        # grad = compute_mt_gradient(x, n, all_diffs, n_jobs=n_jobs)
        grad = compute_mt_gradient_jit(x, n, all_diffs)
        x = sigmoid(grad / tempe)
    return x


def naive_vi(init, n, all_diffs, nm_epochs=6, tempe=1.0, n_jobs=1, dir=None):
    """
    Args:
       init: initializer
        nm_epochs
        tempe: temperature
    """
    if dir is None:
        dir = sys.stdout
    else:
        dir = open(os.path.join(dir, 'log'), 'w+')
    x = init
    for i in tqdm(range(nm_epochs), desc="Epoch"):
        # grad = compute_mt_gradient(x, n, all_diffs, n_jobs=n_jobs)
        start_time = time()
        grad = compute_mt_gradient(x, n, all_diffs)
        x_new = sigmoid(grad / tempe)
        end_time = time()
        print(f"iter={i};diff={np.linalg.norm(np.array(x) - np.array(x_new)) / n};time={(end_time - start_time):0.8f}",
              file=dir)
        x = x_new
    return x


def compute_true_vals_impl(n_player, all_coalition_vals, n_jobs=1, type="all", tempe=0.1):
    """
    Compute true vals via ES.
        n*2^n storage.

    True shapley
    True Banzhaf
    True Mean Field
    """

    # print('Starting ES!')
    # print(n_player)

    all_diffs = compute_all_diffs(n_player, all_coalition_vals)
    # all_coalition_vals =
    log_parti = compute_log_parti(all_coalition_vals, tempe)
    if type == "shap":
        vals_true_shap = compute_true_shapley(n_player, all_diffs)

        error = log_parti - compute_elbo(sigmoid(vals_true_shap / tempe), all_coalition_vals, tempe)
        return vals_true_shap, error

    if type == "banz":
        x = .5 * np.ones(n_player)
        vals_true_banz = compute_mt_gradient(x, n_player, all_diffs)
        error = log_parti - compute_elbo(sigmoid(vals_true_banz / tempe), all_coalition_vals, tempe)
        return vals_true_banz, error

    if type == "fg":
        init = .5 * np.ones(n_player)
        vals_naive_fg = naive_vi(init, n_player, all_diffs, nm_epochs=10 * n_player, tempe=tempe, n_jobs=n_jobs)
        # print(vals_naive_fg)
        error = log_parti - compute_elbo(vals_naive_fg, all_coalition_vals, tempe)
        return vals_naive_fg, error

    vals_true_shap = compute_true_shapley(n_player, all_diffs)
    x = .5 * np.ones(n_player)
    vals_true_banz = compute_mt_gradient(x, n_player, all_diffs)
    init = .5 * np.ones(n_player)
    vals_naive_fg = naive_vi(init, n_player, all_diffs, nm_epochs=10 * n_player, tempe=tempe, n_jobs=n_jobs)

    return all_diffs, vals_true_shap, vals_true_banz, vals_naive_fg
