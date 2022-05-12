import numpy as np
from numba import jit, njit, prange
from sklearn.cluster import KMeans
from tqdm import tqdm


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

    print('All diffs calculated!')
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
    print('Shapley calculated!')
    return vals_true_shap


@jit(nopython=True)
def compute_mt_partial_derivative(i, x, n, all_diffs):
    """
    Args:
        x:  np vector of size n
    """

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


def xlogx(x):
    ''' x:  nx1 vector '''
    n = len(x)
    out = np.zeros((n, 1))
    for i in range(n):
        if 0 == x[i]:
            out[i] = 0
        else:
            out[i] = x[i] * np.log(x[i])

    return out


def compute_mt(x, all_coa_vals, tempe=1.0):
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


def compute_entropy(x):
    """
    Compute the entropy of q(S|x) at x
    """
    minus_x = 1 - x
    return - np.sum(xlogx(x)) - np.sum(xlogx(minus_x))


def compute_elbo(x, all_coa_vals, tempe=1.0):
    return compute_mt(x, all_coa_vals, tempe=tempe) + compute_entropy(x)


def compute_log_parti(all_coa_vals, tempe=1.0):
    return np.log(np.sum(np.exp(np.array(all_coa_vals) / tempe)))


@jit(nopython=True)
def insert0_2loc_i(i, j):
    """insert 0 to location i of j
    """
    all_ones = ~0
    left_mask = (1 << i) - 1
    left = j & left_mask
    right_mask = all_ones << i
    right = (j & right_mask) << 1
    return left | right


@jit(forceobj=True)
def nm_one_bits(j):
    """
    Calc num of one bits in j
    """
    return bin(j).count("1")


@njit(fastmath=True)
def sigmoid(u):
    return 1.0 / (1 + np.exp(-u))


@njit(fastmath=True)
def inv_sigmoid(u):
    return np.log(np.divide(u, (1 - u)))


def clustering(model_name, data, cluster_size, n_jobs=1):
    if model_name == 'kmeans':
        labels = KMeans(n_clusters=cluster_size, n_jobs=n_jobs).fit_predict(data)
    elif model_name == 'rand':
        labels = np.random.permutation(
            np.repeat(np.arange(cluster_size), np.ceil(len(data) / cluster_size))[:len(data)])
    else:
        raise NotImplementedError(f'{model_name} has not implemeted yet')
    return {i: np.where(labels == i)[0] for i in set(labels)}


@jit(nopython=True, parallel=True)
def compute_mt_gradient(x, n, all_diffs):
    # if mc:
    # grad = [compute_mt_partial_derivative_mc(i, x, n, all_diffs, num_samples) for i in prange(n)]
    # else:
    grad = [compute_mt_partial_derivative(i, x, n, all_diffs) for i in prange(n)]

    return np.array(grad)


def mfi(init, n, all_diffs, nm_epochs=6, tempe=1.0, n_jobs=1):
    """
    Args:
       init: initializer
        nm_epochs
        tempe: temperature
    """
    x = init
    for _ in tqdm(range(nm_epochs), desc="Epoch"):
        # grad = compute_mt_gradient(x, n, all_diffs, n_jobs=n_jobs)
        grad = compute_mt_gradient(x, n, all_diffs)
        x = sigmoid(grad / tempe)
    return x


def compute_true_vals_impl(n_player, all_coalition_vals, metric=None, n_jobs=1, tempe=1.0):
    """
    Compute true vals via exhausive search.
        need O(n*2^n) storage.

    True Shapley
    True Banzhaf
    True Variational

    :metric:
        -None: run all three methods
        -shapley: run Shapley
        -banzhaf: run Banzhaf
        -vi: run mean field fixed point iteration
    """

    print('Starting ES!')
    print(n_player)

    results = {}
    all_diffs = compute_all_diffs(n_player, all_coalition_vals)

    if metric is None or 'shapley' in metric:
        results['shapley'] = compute_true_shapley(n_player, all_diffs)
    if metric is None or 'banzhaf' in metric:
        x = .5 * np.ones(n_player)
        results['banzhaf'] = compute_mt_gradient(x, n_player, all_diffs)
        print('Banzhaf calculated!')
    if metric is None or 'vi' in metric:
        init = .5 * np.ones(n_player)
        vi = mfi(init, n_player, all_diffs, nm_epochs=10 * n_player, tempe=tempe, n_jobs=n_jobs)
        results['vi'] = inv_sigmoid(vi)
        print('Variational calculated!')

    return all_diffs, results
