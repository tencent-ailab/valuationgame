import itertools as it
import time
import numpy as np
from utils.utils import xlogx, sigmoid


class parameter():
    '''dummy class to act as struct'''
    pass


def flid_multilinear(x, param):
    #  x must be vector: nx1
    n = param.n
    D = param.D
    minusx = 1 - x

    f = np.asscalar(np.dot(param.u_prime, x)[0])

    for d in np.arange(D):
        xid = x[param.I[:, d]]
        tmp = xid * np.expand_dims(param.Y[:, d], axis=1)
        prod_minusx = np.ones((n, 1))

        #       for l = n-1:-1:1:
        for l in np.arange(n - 2, -1, -1):  # 0-index
            prod_minusx[l] = prod_minusx[l + 1] * minusx[param.I[l + 1, d]]

        f += np.dot(tmp.T, prod_minusx)

    return np.asscalar(f[0])


def flid_elbo(x, param):
    """ x must be vector: nx1
    """
    minusx = 1 - x
    f = flid_multilinear(x, param)
    f = f - np.sum(xlogx(x)) - np.sum(xlogx(minusx))
    return f


def flid_multilinear_gradi(x, i, param):
    """
    calculate i-th derivative
    :param x:
    :param i:
    :param param:
    :return:
    """
    f = flid_multilinear
    xplus = x.copy()
    xplus[i] = 1
    xminus = x.copy()
    xminus[i] = 0
    gradi = f(xplus, param) - f(xminus, param)

    return gradi


def flid_multilinear_grad(x, param):
    """

    :param x:
    :param param:
    :return:
    """
    n = param.n
    f = flid_multilinear
    grad = np.zeros((n, 1))

    for i in np.arange(n):
        xplus = x.copy()
        xplus[i] = 1
        xminus = x.copy()
        xminus[i] = 0
        grad[i] = f(xplus, param) - f(xminus, param)

    return grad


def load_data(dataset_id, D_, n_, data_fig_path):
    #  FLID-0F(V)   F(V)=0  synthetic
    if 1 == dataset_id:

        n = n_
        D = D_
        W = 2 * np.random.rand(n, D)

        # sort according to axis 0
        Y = np.sort(W, axis=0)
        I = np.argsort(W, axis=0)

        sum_u = np.sum(np.sum(Y[:-1, :], axis=0), axis=0)
        #  use uniform u
        u = np.ones([n, 1]) * sum_u / n
        u_prime = u - np.sum(W, axis=1)

        param = parameter()
        param.n = n
        param.D = D
        param.W = W
        param.Y = Y
        param.I = I
        param.u = u
        param.u_prime = u_prime
        param.ub = np.ones([n, 1])
        param.lb = np.zeros([n, 1])

        f = flid_elbo
        grad = flid_multilinear_grad
        gradi = flid_multilinear_gradi
        param.multilinear = flid_multilinear

    # Flid amazon
    elif 2 == dataset_id:
        pass

    return f, grad, gradi, param


def x2marginals(x):
    return x


def solver_double_greedy_dr(f, grad, gradi, param, max_iter):
    a = param.lb
    b = param.ub
    n = param.n

    x = a.copy()
    y = b.copy()
    fsx = []
    fsy = []

    fvx = f(x, param)
    fsx.append(fvx)

    fvy = f(y, param)
    fsy.append(fvy)

    id_seq = np.arange(n)

    t = time.time()
    for i in id_seq:

        fvx = fsx[-1]
        fvy = fsy[-1]
        gradix = gradi(x, i, param)
        ua = sigmoid(gradix)

        gradiy = gradi(y, i, param)
        ub = sigmoid(gradiy)

        xtmp = x.copy()
        xtmp[i] = ua
        delta_a = f(xtmp, param) - fvx

        ytmp = y.copy()
        ytmp[i] = ub
        delta_b = f(ytmp, param) - fvy

        delta_a = max(0, delta_a)
        delta_b = max(0, delta_b)

        if 0 == delta_a and 0 == delta_b:
            ra = 1
        else:
            ra = delta_a / (delta_a + delta_b)

        u = ra * ua + (1 - ra) * ub
        x[i] = u
        y[i] = u
        fvx = f(x, param)
        fvy = f(y, param)
        fsx.append(fvx)
        fsy.append(fvy)

    run_time = time.time() - t

    opt_f = fsx[-1]
    fs = fsx
    opt_x = x
    margs = x2marginals(opt_x)

    return opt_x, opt_f, fs, margs, run_time


def solver_double_greedy_dr_multiepoch(f, grad, gradi, param, num_epoch):
    a = param.lb
    b = param.ub
    n = param.n

    x = a.copy()
    y = b.copy()

    fsx = []
    fsy = []

    fvx = f(x, param)
    fsx.append(fvx)
    fvy = f(y, param)
    fsy.append(fvy)

    id_seq = np.arange(n)

    t = time.time()
    for i in id_seq:

        fvx = fsx[-1]
        fvy = fsy[-1]
        gradix = gradi(x, i, param)
        ua = sigmoid(gradix)

        gradiy = gradi(y, i, param)
        ub = sigmoid(gradiy)

        xtmp = x.copy()
        xtmp[i] = ua
        delta_a = f(xtmp, param) - fvx

        ytmp = y.copy()
        ytmp[i] = ub
        delta_b = f(ytmp, param) - fvy

        delta_a = max(0, delta_a)
        delta_b = max(0, delta_b)

        if 0 == delta_a and 0 == delta_b:
            ra = 1
        else:
            ra = delta_a / (delta_a + delta_b)

        u = ra * ua + (1 - ra) * ub
        x[i] = u
        y[i] = u
        fvx = f(x, param)
        fvy = f(y, param)
        fsx.append(fvx)
        fsy.append(fvy)

    for _ in np.arange(num_epoch):
        for i in id_seq:
            fvx = fsx[-1]
            gradix = gradi(x, i, param)
            ua = sigmoid(gradix)

            x[i] = ua
            delta_a = f(x, param) - fvx

            fvx = delta_a + fvx
            fsx.append(fvx)

    run_time = time.time() - t
    opt_f = fsx[-1]
    fs = fsx
    opt_x = x
    margs = x2marginals(opt_x)

    return opt_x, opt_f, fs, margs, run_time


def solver_shapley(f, grad, gradi, param, max_iter):
    n = param.n
    fmultilinear = param.multilinear

    t = time.time()
    weights = np.zeros((n, 1))
    for s in range(n):
        weights[s] = np.math.factorial(s) * np.math.factorial(n - s - 1) / np.math.factorial(n)

    opt_x = np.zeros([n, 1])
    for i in range(n):
        # compute ith true marginals
        g = [(0, 1) for _ in range(n)]
        g[i] = (0,)
        powerset = it.product(*g)
        ri = 0
        for ss in powerset:
            x = np.array(ss)[:, None]
            FS = fmultilinear(x, param)
            x[i] = 1
            FSi = fmultilinear(x, param)
            ri += weights[sum(ss)] * (FSi - FS)

        opt_x[i] = ri

    margs = sigmoid(opt_x)
    opt_f = fmultilinear(np.ones([n, 1]), param) - fmultilinear(np.zeros([n, 1]), param)
    fs = None
    run_time = time.time() - t
    return opt_x, opt_f, fs, margs, run_time


def solver_banzhaf(f, grad, gradi, param, max_iter):
    n = param.n
    fmultilinear = param.multilinear

    t = time.time()
    weights = np.ones((n, 1)) / (2 ** (n - 1))

    opt_x = np.zeros([n, 1])
    for i in range(n):
        # compute ith true marginals
        g = [(0, 1) for _ in range(n)]
        g[i] = (0,)
        powerset = it.product(*g)
        ri = 0
        for ss in powerset:
            x = np.array(ss)[:, None]
            FS = fmultilinear(x, param)
            x[i] = 1
            FSi = fmultilinear(x, param)
            ri += weights[sum(ss)] * (FSi - FS)
        opt_x[i] = ri

    margs = sigmoid(opt_x)
    opt_f = fmultilinear(np.ones([n, 1]), param) - fmultilinear(np.zeros([n, 1]), param)
    fs = None
    run_time = time.time() - t
    # print('Banzhaf')
    return opt_x, opt_f, fs, margs, run_time


def solver_single_greedy_1(f, grad, gradi, param, max_iter):
    # a = param.lb
    b = param.ub
    n = param.n

    x = b.copy()
    fsx = []

    fvx = f(x, param)
    fsx.append(fvx)
    id_seq = np.arange(n)

    t = time.time()
    for i in id_seq:
        fvx = fsx[-1]
        gradix = gradi(x, i, param)
        ua = sigmoid(gradix)

        x[i] = ua
        delta_a = f(x, param) - fvx

        fvx = delta_a + fvx
        fsx.append(fvx)

    run_time = time.time() - t

    opt_f = fsx[-1]
    fs = fsx
    opt_x = x
    margs = x2marginals(opt_x)

    return opt_x, opt_f, fs, margs, run_time


def solver_double_greedy_submodular(f, grad, gradi, param, max_iter):
    a = param.lb
    b = param.ub
    n = param.n

    x = a.copy()
    y = b.copy()

    fsx = []
    fsy = []

    fvx = f(x, param)
    fsx.append(fvx)

    fvy = f(y, param)
    fsy.append(fvy)
    id_seq = np.arange(n)

    t = time.time()
    for i in id_seq:

        fvx = fsx[-1]
        fvy = fsy[-1]
        gradix = gradi(x, i, param)
        ua = sigmoid(gradix)

        gradiy = gradi(y, i, param)
        ub = sigmoid(gradiy)

        xtmp = x.copy()
        xtmp[i] = ua
        delta_a = f(xtmp, param) - fvx

        ytmp = y.copy()
        ytmp[i] = ub
        delta_b = f(ytmp, param) - fvy

        if delta_a >= delta_b:
            x[i] = ua;
            y[i] = ua
            fvx = fvx + delta_a
            fsx.append(fvx)
            fsy.append(f(y, param))
        else:
            y[i] = ub
            x[i] = ub
            fvy = fvy + delta_b
            fsy.append(fvy)
            fsx.append(f(x, param))

    run_time = time.time() - t

    opt_f = fsx[-1]
    fs = fsx
    opt_x = x
    margs = x2marginals(opt_x)

    return opt_x, opt_f, fs, margs, run_time


def solver_double_greedy_submodular_multiepoch(f, grad, gradi, param, num_epoch):
    a = param.lb
    b = param.ub
    n = param.n
    x = a.copy()
    y = b.copy()

    fsx = []
    fsy = []

    fvx = f(x, param)
    fsx.append(fvx)

    fvy = f(y, param)
    fsy.append(fvy)
    id_seq = np.arange(n)

    t = time.time()
    for i in id_seq:

        fvx = fsx[-1]
        fvy = fsy[-1]
        gradix = gradi(x, i, param)
        ua = sigmoid(gradix)

        gradiy = gradi(y, i, param)
        ub = sigmoid(gradiy)

        xtmp = x.copy()
        xtmp[i] = ua
        delta_a = f(xtmp, param) - fvx

        ytmp = y.copy()
        ytmp[i] = ub
        delta_b = f(ytmp, param) - fvy

        if delta_a >= delta_b:
            x[i] = ua
            y[i] = ua
            fvx = fvx + delta_a
            fsx.append(fvx)
            fsy.append(f(y, param))
        else:
            y[i] = ub
            x[i] = ub
            fvy = fvy + delta_b
            fsy.append(fvy)
            fsx.append(f(x, param))

    for _ in range(num_epoch):
        for i in id_seq:
            gradix = gradi(x, i, param)
            ua = sigmoid(gradix)

            x[i] = ua
            delta_a = f(x, param) - fvx

            fvx = delta_a + fvx
            fsx.append(fvx)

    run_time = time.time() - t

    opt_f = fsx[-1]
    fs = fsx
    opt_x = x
    margs = x2marginals(opt_x)

    return opt_x, opt_f, fs, margs, run_time


def solver_single_greedy_05_multiepoch(f, grad, gradi, param, num_epoch):
    # a = param.lb
    # b = param.ub
    n = param.n

    x = np.ones([n, 1]) * 0.5
    fsx = []

    fvx = f(x, param)
    fsx.append(fvx)
    id_seq = np.arange(n)

    t = time.time()
    for i in id_seq:
        fvx = fsx[-1]
        gradix = gradi(x, i, param)
        ua = sigmoid(gradix)

        x[i] = ua
        delta_a = f(x, param) - fvx

        fvx = delta_a + fvx
        fsx.append(fvx)

    for _ in range(num_epoch):
        for i in id_seq:
            gradix = gradi(x, i, param)
            ua = sigmoid(gradix)

            x[i] = ua
            delta_a = f(x, param) - fvx

            fvx += delta_a
            fsx.append(fvx)

    run_time = time.time() - t

    opt_f = fsx[-1]
    fs = fsx
    opt_x = x
    margs = x2marginals(opt_x)

    return opt_x, opt_f, fs, margs, run_time


def gradient_ascent(f, grad, gradi, param, num_epoch):
    a = param.lb
    b = param.ub

    x = (a + b) / 2
    x = x.copy()
    fsx = []

    fvx = f(x, param)
    fsx.append(fvx)

    t = time.time()

    for _ in range(num_epoch):
        gradx = grad(x, param)
        x = sigmoid(gradx)
        fvx = f(x, param)
        fsx.append(fvx)

    run_time = time.time() - t

    opt_f = fsx[-1]
    fs = fsx
    opt_x = x
    margs = x2marginals(opt_x)

    return opt_x, opt_f, fs, margs, run_time


def solver_ground_truth_es(f, grad, gradi, param, max_iter):
    """
    using exhausitive search
    :param f:
    :param grad:
    :param gradi:
    :param param:
    :param max_iter:
    :return:
    """

    n = param.n
    fmultilinear = param.multilinear

    t = time.time()
    Z = 0
    ground = [(0, 1) for _ in range(n)]
    powerset = it.product(*ground)

    for dd in powerset:
        x = np.array(dd)[:, np.newaxis]
        Z = Z + np.exp(fmultilinear(x, param))

    run_time = time.time() - t
    opt_f = np.log(Z)

    fs = opt_f
    opt_x = np.zeros([n, 1])
    for i in range(n):
        # compute ith true marginals
        g = [(0, 1) for _ in range(n)]
        g[i] = (1,)
        powerset = it.product(*g)
        ithZ = 0
        #        print(list(powerset))
        for ss in powerset:
            x = np.array(ss)[:, None]
            ithZ += np.exp(fmultilinear(x, param))
        opt_x[i] = ithZ / Z
    margs = opt_x

    return opt_x, opt_f, fs, margs, run_time


def solver_ground_truth_flid(f, grad, gradi, param, max_iter):
    """
    complexity:  O(n^(D+1))
    :param f:
    :param grad:
    :param gradi:
    :param param:
    :param max_iter:
    :return:
    """

    n = param.n;
    D = param.D

    u_prime = param.u_prime

    t = time.time()
    I = param.I
    W = param.W
    Z = 0

    id_seq = list(range(n))

    ground = [id_seq for _ in range(D)]
    powerset = it.product(*ground)

    for dd in powerset:

        idI = dd
        idW = np.zeros([D, 1], dtype=int)

        for j in range(D):
            idW[j] = I[idI[j], j]

        II = np.unique(idW)

        X = []
        for j in range(D):
            X = np.union1d(X, (I[idI[j] + 1:, j]))

        if np.intersect1d(II, X).size > 0:
            continue

        tmp_sum = 0
        for d in range(D):
            tmp_sum = tmp_sum + W[idW[d], d]

        for i in range(len(II)):
            tmp_sum = tmp_sum + u_prime[II[i]]

        Vprime = np.setdiff1d(id_seq, II)
        Vprime = np.setdiff1d(Vprime, X)

        tmp_prod = 1
        if Vprime.size > 0:
            for i in range(len(Vprime)):
                tmp_prod = tmp_prod * (1 + np.exp(u_prime[Vprime[i]]))

        Z = Z + (np.exp(tmp_sum) * tmp_prod)[0]

    Z = Z + 1;

    run_time = time.time() - t

    opt_f = np.log(Z)
    fs = opt_f
    opt_x = np.zeros([n, 1])
    fm = param.multilinear
    # print('z:', Z)

    for i in range(n):
        xtmp = np.zeros([n, 1])
        xtmp[i] = 1
        opt_x[i] = fm(xtmp, param) / Z

    margs = opt_x

    return opt_x, opt_f, fs, margs, run_time


def launch_solver(f, grad, gradi, param, method, num_epoch):
    func_names = (
        'solver_double_greedy_dr',  # % 0
        'solver_shapley',  # 1
        'solver_banzhaf',  # 2
        'solver_banzhaf_025',  # 3
        'solver_double_greedy_dr_multiepoch',  # 4
        'solver_single_greedy_05_multiepoch',  # 5
        'gradient_ascent',  # 6
        'solver_ground_truth_flid',  # 7
        'solver_ground_truth_es',  # 8
        'solver_single_greedy_0_multiepoch_momentum',  # 9
        'solver_double_greedy_submodular_multiepoch'  # 10
    )

    func = eval(func_names[method])

    [x_opt, opt_f, fs, margs, runtime] \
        = func(f, grad, gradi, param, num_epoch)

    return x_opt, opt_f, fs, margs, runtime
