# MIT License
#
# Copyright (c) 2019 amiratag

import inspect
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import logistic
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score


def convergence_plots(marginals):
    plt.rcParams['figure.figsize'] = 15, 15
    for i, idx in enumerate(np.arange(min(25, marginals.shape[-1]))):
        plt.subplot(5, 5, i + 1)
        plt.plot(np.cumsum(marginals[:, idx]) / np.arange(1, len(marginals) + 1))


def is_integer(array):
    return (np.equal(np.mod(array, 1), 0).mean() == 1)


def is_fitted(model):
    """Checks if model object has any attributes ending with an underscore"""
    return 0 < len([k for k, v in inspect.getmembers(model) if k.endswith('_') and not k.startswith('__')])


def get_model(mode, **kwargs):
    if mode == 'logistic':
        solver = kwargs.get('solver', 'liblinear')
        n_jobs = kwargs.get('n_jobs', None)
        max_iter = kwargs.get('max_iter', 5000)

        model = LogisticRegression(solver=solver, n_jobs=n_jobs,
                                   max_iter=max_iter, random_state=666)

    elif mode == 'linear':
        model = LinearRegression(random_state=666)

    else:
        raise ValueError("Invalid mode!")
    return model


def generate_features(latent, dependency):
    #    print('latent: {0}, dependency: {1}'.format(latent.shape , dependency))

    features = []
    n = latent.shape[0]
    exp = latent
    holder = latent
    for order in range(1, dependency + 1):
        features.append(np.reshape(holder, [n, -1]))
        exp = np.expand_dims(exp, -1)
        holder = exp * np.expand_dims(holder, 1)

    return np.concatenate(features, axis=-1)


def label_generator(problem, X, param, difficulty=1, beta=None, important=None):
    """
    @important: # important dims
    @difficulty: order of polynomial
    @beta: 
    """
    if important is None or important > X.shape[-1]:
        important = X.shape[-1]

    dim_latent = sum([important ** i for i in range(1, difficulty + 1)])
    # print('dim_latent:', dim_latent)

    if beta is None:
        beta = np.random.normal(size=[1, dim_latent])

    important_dims = np.random.choice(X.shape[-1], important, replace=False)
    funct_init = lambda inp: np.sum(beta * generate_features(inp[:, important_dims], difficulty), -1)

    batch_size = max(100, min(len(X), 10000000 // dim_latent))
    y_true = np.zeros(len(X))

    while True:
        try:
            for itr in range(int(np.ceil(len(X) / batch_size))):
                y_true[itr * batch_size: (itr + 1) * batch_size] = funct_init(
                    X[itr * batch_size: (itr + 1) * batch_size])
            break
        except MemoryError:
            batch_size = batch_size // 2

    mean, std = np.mean(y_true), np.std(y_true)
    funct = lambda x: (np.sum(beta * generate_features(
        x[:, important_dims], difficulty), -1) - mean) / std
    y_true = (y_true - mean) / std

    if problem is 'classification':
        y_true = logistic.cdf(param * y_true)
        y = (np.random.random(X.shape[0]) < y_true).astype(int)
    elif problem is 'regression':
        y = y_true + param * np.random.normal(size=len(y_true))
    else:
        raise ValueError('Invalid problem specified!')

    #    print('funct:', funct)
    return beta, y, y_true, funct


def one_iteration(clf, X, y, X_test, y_test, mean_score, tol=0.0, c=None, metric='accuracy'):
    """Runs one iteration of TMC-Shapley."""

    if metric == 'auc':
        def score_func(clf, a, b):
            return roc_auc_score(b, clf.predict_proba(a)[:, 1])
    elif metric == 'accuracy':
        def score_func(clf, a, b):
            return clf.score(a, b)
    else:
        raise ValueError("Wrong metric!")
    if c is None:
        c = {i: np.array([i]) for i in range(len(X))}
    idxs, marginal_contribs = np.random.permutation(len(c.keys())), np.zeros(len(X))
    new_score = np.max(np.bincount(y)) * 1. / len(y) if np.mean(y // 1 == y / 1) == 1 else 0.
    start = 0
    if start:
        X_batch, y_batch = \
            np.concatenate([X[c[idx]] for idx in idxs[:start]]), np.concatenate([y[c[idx]] for idx in idxs[:start]])
    else:
        X_batch, y_batch = np.zeros((0,) + tuple(X.shape[1:])), np.zeros(0).astype(int)
    for n, idx in enumerate(idxs[start:]):
        try:
            clf = clone(clf)
        except:
            clf.fit(np.zeros((0,) + X.shape[1:]), y)
        old_score = new_score
        X_batch, y_batch = np.concatenate([X_batch, X[c[idx]]]), np.concatenate([y_batch, y[c[idx]]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                clf.fit(X_batch, y_batch)
                temp_score = score_func(clf, X_test, y_test)
                if temp_score > -1 and temp_score < 1.:  # Removing measningless r2 scores
                    new_score = temp_score
            except:
                continue
        marginal_contribs[c[idx]] = (new_score - old_score) / len(c[idx])
        if np.abs(new_score - mean_score) / mean_score < tol:
            break
    return marginal_contribs, idxs


def marginals(clf, X, y, X_test, y_test, c=None, tol=0., trials=3000, mean_score=None, metric='accuracy'):
    if metric == 'auc':
        def score_func(clf, a, b):
            return roc_auc_score(b, clf.predict_proba(a)[:, 1])
    elif metric == 'accuracy':
        def score_func(clf, a, b):
            return clf.score(a, b)
    else:
        raise ValueError("Wrong metric!")
    if mean_score is None:
        accs = []
        for _ in range(100):
            bag_idxs = np.random.choice(len(y_test), len(y_test))
            accs.append(score_func(clf, X_test[bag_idxs], y_test[bag_idxs]))
        mean_score = np.mean(accs)
    marginals, idxs = [], []
    for trial in range(trials):
        if 10 * (trial + 1) / trials % 1 == 0:
            print('{} out of {}'.format(trial + 1, trials))
        marginal, idx = one_iteration(clf, X, y, X_test, y_test, mean_score, tol=tol, c=c, metric=metric)
        marginals.append(marginal)
        idxs.append(idx)
    return np.array(marginals), np.array(idxs)


def early_stopping(marginals, idxs, stopping):
    stopped_marginals = np.zeros_like(marginals)
    for i in range(len(marginals)):
        stopped_marginals[i][idxs[i][:stopping]] = marginals[i][idxs[i][:stopping]]
    return np.mean(stopped_marginals, 0)


def error(mem):
    """
    
    """
    if len(mem) < 100:
        return 1.0
    all_vals = (np.cumsum(mem, 0) / np.reshape(np.arange(1, len(mem) + 1), (-1, 1)))[-100:]
    errors = np.mean(np.abs(all_vals[-100:] - all_vals[-1:]) / (np.abs(all_vals[-1:]) + 1e-12), -1)
    return np.max(errors)


def my_accuracy_score(clf, X, y):
    probs = clf.predict_proba(X)
    predictions = np.argmax(probs, -1)
    return np.mean(np.equal(predictions, y))


def my_f1_score(clf, X, y):
    predictions = clf.predict(X)
    if len(set(y)) == 2:
        return f1_score(y, predictions)
    return f1_score(y, predictions, average='macro')


def my_auc_score(clf, X, y):
    probs = clf.predict_proba(X)
    true_probs = probs[np.arange(len(y)), y]
    return roc_auc_score(y, true_probs)


def my_xe_score(clf, X, y):
    probs = clf.predict_proba(X)
    true_probs = probs[np.arange(len(y)), y]
    true_log_probs = np.log(np.clip(true_probs, 1e-12, None))
    return np.mean(true_log_probs)