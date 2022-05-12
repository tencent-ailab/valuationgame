# MIT License
#
# Copyright (c) 2019 amiratag
import matplotlib
matplotlib.use('Agg')
import os
import tensorflow as tf
import numpy as np
from sklearn.base import clone
import matplotlib.pyplot as plt
import warnings
import _pickle as pkl
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
from tqdm import tqdm

from data_valuation.dshap_utils import sigmoid, compute_elbo, compute_true_vals_impl, compute_log_parti
from third_party.data_shap_utils import my_auc_score, my_xe_score, get_model


class DShap(object):

    def __init__(self, X, y, X_test, y_test, num_train, num_test, sources=None, directory=None,
                 problem='classification', model_family='logistic', metric='accuracy',
                 seed=None, **kwargs):
        """
        Args:
            X: Data covariates
            y: Data labels
            X_test: Test+Held-out covariates
            y_test: Test+Held-out labels
            sources: An array or dictionary assigning each point to its group.
                If None, every points gets its individual value.
                
            num_test: Number of data points used for evaluation metric.
            directory: Directory to save results and figures.
            problem: "Classification" or "Regression"(Not implemented yet.)
            model_family: The model family used for learning algorithm
            metric: Evaluation metric
            seed: Random seed. When running parallel monte-carlo samples,
                we initialize each with a different seed to prevent getting 
                same permutations.
            **kwargs: Arguments of the model
        """

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)
        self.problem = problem
        self.model_family = model_family
        self.metric = metric
        self.directory = directory
        self.hidden_units = kwargs.get('hidden_layer_sizes', [])
        if self.model_family is 'logistic':
            self.hidden_units = []
        if self.directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)
                os.makedirs(os.path.join(directory, 'weights'))
                os.makedirs(os.path.join(directory, 'plots'))
            self._initialize_instance(X, y, X_test, y_test, num_train, num_test, sources)

        if len(set(self.y)) > 2:
            assert self.metric != 'f1' and self.metric != 'auc', 'Invalid metric!'
        is_regression = (np.mean(self.y // 1 == self.y) != 1)
        is_regression = is_regression or isinstance(self.y[0], np.float32)
        self.is_regression = is_regression or isinstance(self.y[0], np.float64)
        self.model = get_model(self.model_family, **kwargs)
        self.random_score = self.init_score(self.metric)
        self.print_freq = 2  # number of prints during run
        self.all_coalition_vals = None
        self.all_diffs = None

    def _initialize_instance(self, X, y, X_test, y_test, num_train, num_test, sources=None):
        """Loads or creates data."""

        if sources is None:
            sources = {i: np.array([i]) for i in range(len(X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}

        data_dir = os.path.join(self.directory, 'data.pkl')
        if os.path.exists(data_dir):
            data_dic = pkl.load(open(data_dir, 'rb'))
            self.X_heldout, self.y_heldout = data_dic['X_heldout'], data_dic['y_heldout']
            self.X_test, self.y_test = data_dic['X_test'], data_dic['y_test']
            self.X, self.y = data_dic['X'], data_dic['y']
            self.sources = data_dic['sources']
        else:
            self.X_heldout, self.y_heldout = X_test[:-num_test], y_test[:-num_test]
            # last num_test points as test data
            self.X_test, self.y_test = X_test[-num_test:], y_test[-num_test:]
            if num_train > 0:
                self.X, self.y = X[:num_train], y[:num_train]
                sources = {i: [_ for _ in v if _ in range(num_train)] for i, v in sources.items()}
                self.sources = {i: v for i, v in sources.items() if len(v) > 0}
            else:
                self.X, self.y, self.sources = X, y, sources

            pkl.dump({'X': self.X, 'y': self.y, 'X_test': self.X_test,
                      'y_test': self.y_test, 'X_heldout': self.X_heldout,
                      'y_heldout': self.y_heldout, 'sources': self.sources},
                     open(data_dir, 'wb'))

        # loo_dir = os.path.join(self.directory, 'loo.pkl')
        # self.vals_loo = None
        # if os.path.exists(loo_dir):
        #     self.vals_loo = pkl.load(open(loo_dir, 'rb'))['loo']
        # previous_results = os.listdir(self.directory)
        # tmc_numbers = [int(name.split('.')[-2].split('_')[-1])
        #                for name in previous_results if 'mem_tmc' in name]
        #
        # g_numbers = [int(name.split('.')[-2].split('_')[-1])
        #              for name in previous_results if 'mem_g' in name]
        # self.tmc_number = str(0) if len(g_numbers) == 0 else str(np.max(tmc_numbers) + 1)
        # self.g_number = str(0) if len(g_numbers) == 0 else str(np.max(g_numbers) + 1)
        # tmc_dir = os.path.join(self.directory, 'mem_tmc_{}.pkl'.format(self.tmc_number.zfill(4)))
        # g_dir = os.path.join(self.directory, 'mem_g_{}.pkl'.format(self.g_number.zfill(4)))
        # 
        # self.mem_tmc, self.mem_g = [np.zeros((0, self.X.shape[0])) for _ in range(2)]
        #
        # idxs_shape = (0, self.X.shape[0] if self.sources is None else len(self.sources.keys()))
        # self.idxs_tmc, self.idxs_g = [np.zeros(idxs_shape).astype(int) for _ in range(2)]
        # pkl.dump({'mem_tmc': self.mem_tmc, 'idxs_tmc': self.idxs_tmc}, open(tmc_dir, 'wb'))
        #
        # if self.model_family not in ['logistic', 'NN']:
        #     return
        # pkl.dump({'mem_g': self.mem_g, 'idxs_g': self.idxs_g}, open(g_dir, 'wb'))

    def init_score(self, metric):
        """ 
        Gives the value of an initial untrained model.
        """

        if metric == 'accuracy':
            # ratio of class with more labels
            #            return np.max(np.bincount(self.y_test).astype(float)/len(self.y_test))
            return 0.5
        if metric == 'f1':
            return np.mean([f1_score(
                self.y_test, np.random.permutation(self.y_test)) for _ in range(1000)])
        if metric == 'auc':
            return 0.5

        random_scores = []
        for _ in range(100):
            self.model.fit(self.X, np.random.permutation(self.y))
            random_scores.append(self.value(self.model, metric))

        return np.mean(random_scores)

    def value(self, model, metric=None, X=None, y=None):
        """Computes the values of the given model.
        
        Args:
            model: The model to be evaluated.
            metric: Valuation metric. If None the object's default
                metric is used.
            X: Covariates, valuation is performed on a data different from test set.
            y: Labels, if valuation is performed on a data different from test set.
        """
        if metric is None:
            metric = self.metric
        if X is None:
            X = self.X_test
        if y is None:
            y = self.y_test
        if metric == 'accuracy':
            return model.score(X, y)
        if metric == 'f1':
            assert len(set(y)) == 2, 'Data has to be binary for f1 metric.'
            return f1_score(y, model.predict(X))

        if metric == 'auc':
            assert len(set(y)) == 2, 'Data has to be binary for auc metric.'
            return my_auc_score(model, X, y)

        if metric == 'xe':
            return my_xe_score(model, X, y)

        raise ValueError('Invalid metric!')

    def get_coalition_val(self, subset, metric):
        """Get the value of a coalition.
        
        Args:
            subset: a non-empty list
        """
        if not subset or len(subset) < 2:
            # contribution of a single data point is 0 for binary classification
            return self.init_score(metric)

        self.restart_model()
        data_ids = np.concatenate([self.sources[i] for i in subset])
        #        print(data_ids)
        X, y = self.X[data_ids], self.y[data_ids]
        if len(set(y)) < 2:  # if there is only one class, no contribution
            return self.init_score(metric)

        self.model.fit(X, y)
        return self.value(self.model, metric=metric)

    def compute_all_coalition_vals(self, metric, n_player, n_jobs=1):
        """Will only be run once."""
        if self.all_coalition_vals is not None:
            return
            # all_vals = {}
        # for i in tqdm(range(2**self.n)):
        #    subset = []
        #
        #    for k in range(self.n):
        #        if i & 1<<k: subset.append(k)
        #
        #    all_vals[int(i)] = self.get_coalition_val(subset, metric)
        # self.all_coalition_vals = all_vals
        all_vals = Parallel(n_jobs=n_jobs, backend="multiprocessing")(delayed(self.get_coalition_val)
                                                                      ([num for num, x in
                                                                        enumerate(bin(i)[2:].zfill(n_player)) if
                                                                        x == '1'], metric)
                                                                      for i in tqdm(range(2 ** n_player)))

        # self.all_coalition_vals = {i: val for i, val in enumerate(all_vals)}
        self.all_coalition_vals = all_vals
        print('All coalition values computed.')

    def compute_true_vals(self, sources=None, metric=None, val_criterion=None, tempe=1.0, n_jobs=1):
        """
        Compute true vals via ES.  
            n*2^n storage.

        True Shapley
        True Banzhaf
        True Variational
        """
        self.is_mc = False
        if sources is None and self.sources is None:
            self.sources = {i: np.array([i]) for i in range(len(self.X))}
        elif sources is not None and isinstance(sources, (list, tuple)):
            self.sources = {i: np.where(sources == i)[0] for i in set(sources)}
        elif sources is not None and isinstance(sources, dict):
            self.sources = sources

        self.val_criterion = val_criterion
        if val_criterion is None:
            self.val_criterion = ['vi', 'shapley', 'banzhaf', 'random']

        print('Starting ES!')
        if metric is None:
            metric = self.metric
        n_player = len(self.sources)  # num of players
        # print(n_player)

        # output of this function is : list of 2^{n_player}
        self.compute_all_coalition_vals(metric, n_player, n_jobs=n_jobs)

        self.log_parti = compute_log_parti(self.all_coalition_vals, tempe=tempe)

        self.all_diffs, self.results = compute_true_vals_impl(n_player, self.all_coalition_vals,
                                                              metric=val_criterion,
                                                              tempe=tempe,
                                                              n_jobs=n_jobs)
        if 'shapley' in self.results:
            print(
                f"error of vals_true_shap: {self.log_parti - compute_elbo(sigmoid(self.results['shapley'] / tempe), self.all_coalition_vals, tempe=tempe)}")
        if 'banzhaf' in self.results:
            print(
                f"error of vals_true_banz: {self.log_parti - compute_elbo(sigmoid(self.results['banzhaf'] / tempe), self.all_coalition_vals, tempe=tempe)}")
        if 'vi' in self.results:
            print(
                f"error of vals_naive_vi: {self.log_parti - compute_elbo(sigmoid(self.results['vi']), self.all_coalition_vals, tempe=tempe)}")

    def restart_model(self):

        try:
            self.model = clone(self.model)  # deep copy params, but no data
        except:
            self.model.fit(np.zeros((0,) + self.X.shape[1:]), self.y)

    def performance_plots(self, vals, tempe=1, name=None, num_plot_markers=20,
                          sources=None, val_criterion=None, percent=1.0):
        """Plots the effect of removing valuable points.
        
        Args:
            vals: A list of different valuations of data points each
                 in the format of an array in the same length of the data.
            name: Name of the saved plot if not None.
            num_plot_markers: number of points in each plot.
            sources: If values are for sources of data points rather than
                   individual points. In the format of an assignment array
                   or dict.
            val_criterion: Only plot subset of val criteria. including
                         vi, shapley, banzhaf
                   
        Returns:
            Plots showing the change in performance as points are removed
            from most valuable to least.
        """
        if self.directory is not None and name is not None:
            plt.clf()
        plt.rcParams['figure.figsize'] = 8, 6
        plt.rcParams['font.size'] = 20
        plt.xlabel('Fraction of training data removed (%)')
        plt.ylabel('Test accuracy (%)', fontsize=20)

        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.sources))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}

        vals_sources = {name: np.array([np.sum(val[sources[i]]) for i in range(len(sources.keys()))])
                        for name, val in vals.items()}
        #        print(vals_sources)
        if val_criterion is None:
            val_criterion = ['vi', 'shapley', 'banzhaf', 'random']

        if len(sources.keys()) < num_plot_markers:
            num_plot_markers = len(sources.keys())

        plot_points = np.arange(0, max(len(sources.keys()) - 10, num_plot_markers),
                                max(len(sources.keys()) // num_plot_markers, 1))

        perfs = {name: self._portion_performance(
            np.argsort(vals_source)[::-1], plot_points, sources=self.sources)
            for name, vals_source in vals_sources.items()}

        rnd = np.mean([self._portion_performance(
            np.random.permutation(np.arange(len(list(vals_sources.values())[0]))),
            plot_points, sources=self.sources) for _ in range(10)], axis=0)

        legends = []
        percent_int = int(len(plot_points) * percent)
        if 'vi' in val_criterion and 'vi' in self.val_criterion:
            plt.plot(plot_points[:percent_int] / len(self.sources) * 100, perfs['vi'][:percent_int] * 100, '-', lw=5,
                     ms=10, color='black')
            #            legends = ['TMC-Shapley ', 'True-Shapley ', 'True-Banzhaf', 'Random']
            if self.is_mc:
                legends.append(f"Variational Index")
            else:
                vi_err = self.log_parti - compute_elbo(sigmoid(self.results['vi']), self.all_coalition_vals,
                                                       tempe=tempe)
                legends.append(f"Variational Index ({vi_err:.7f})")

        if 'random' in val_criterion and 'random' in self.val_criterion:
            plt.plot(plot_points[:percent_int] / len(self.sources) * 100, rnd[:percent_int] * 100, ':', lw=5, ms=10,
                     color='r')
            legends.append('Random')
        if 'shapley' in val_criterion and 'shapley' in self.val_criterion:
            plt.plot(plot_points[:percent_int] / len(self.sources) * 100, perfs['shapley'][:percent_int] * 100, '--',
                     lw=5, ms=10, color='orange')
            #            legends = ['TMC-Shapley ', 'True-Shapley ', 'True-Banzhaf', 'Random']
            if self.is_mc:
                legends.append(f'True-Shapley')
            else:
                shap_err = self.log_parti - compute_elbo(sigmoid(self.results['shapley'] / tempe),
                                                         self.all_coalition_vals, tempe=tempe)
                legends.append(f'True-Shapley ({shap_err:.7f})')

        if 'banzhaf' in val_criterion and 'banzhaf' in self.val_criterion:
            plt.plot(plot_points[:percent_int] / len(self.sources) * 100, perfs['banzhaf'][:percent_int] * 100, '-.',
                     lw=5, ms=10, color='g')
            if self.is_mc:
                legends.append(f'True-Banzhaf')
            else:
                banz_err = self.log_parti - compute_elbo(sigmoid(self.results['banzhaf'] / tempe),
                                                         self.all_coalition_vals, tempe=tempe)
                legends.append(f'True-Banzhaf ({banz_err:.7f})')
        plt.legend(legends, frameon=False, fontsize=14)

        if self.directory is not None and name is not None:
            plt.savefig(os.path.join(self.directory, 'plots', '{}.pdf'.format(name)),
                        format='pdf',
                        bbox_inches='tight')
            plt.savefig(os.path.join(self.directory, 'plots', '{}.png'.format(name)),
                        format='png',
                        bbox_inches='tight')
            pkl.dump([perfs, rnd, self.all_diffs, self.results], open(os.path.join(self.directory, 'perfs.pkl'), 'wb'))

        #            plt.close()
        plt.show()

    def _portion_performance(self, idxs, plot_points, sources=None):
        """
        Given a set of indexes, starts removing points from the first element
           and evaluates the new model after removing each point.
        """
        if sources is None:
            sources = {i: np.array([i]) for i in range(len(self.X))}
        elif not isinstance(sources, dict):
            sources = {i: np.where(sources == i)[0] for i in set(sources)}

        scores = []
        init_score = self.random_score
        for i in range(len(plot_points), 0, -1):
            keep_idxs = np.concatenate([sources[idx] for idx in idxs[plot_points[i - 1]:]], -1)
            X_batch, y_batch = self.X[keep_idxs], self.y[keep_idxs]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # need to include all labels
                if (self.is_regression or len(set(y_batch)) == len(set(self.y_test))) and (len(X_batch) > 1):
                    self.restart_model()
                    self.model.fit(X_batch, y_batch)

                    # evaluate on a heldout dataset
                    #                    scores.append(self.value(self.model, metric=self.metric,
                    #                                             X=self.X_heldout, y=self.y_heldout))

                    # evaluate on the test dataset
                    scores.append(self.value(self.model, metric=self.metric,
                                             X=self.X_test, y=self.y_test))

                else:
                    scores.append(init_score)

        return np.array(scores)[::-1]
