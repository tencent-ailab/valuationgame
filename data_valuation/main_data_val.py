import argparse
import os
import shutil

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split

from data_valuation.dshap import DShap
from data_valuation.dshap_utils import clustering
from third_party.data_shap_utils import get_model, label_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', '-p',
                        type=str, choices=['classification', 'regression'], default='classification',
                        help="The problem setting")
    parser.add_argument('--val_criterion', '-s',
                        type=str, nargs='+', choices=['all', 'vi', 'shapley', 'banzhaf', 'random'], default=['all'],
                        help="the valuation criterion to be computed")
    parser.add_argument('--model', '-m',
                        type=str, choices=['logistic', 'linear'], default='logistic',
                        help="The classification model")
    parser.add_argument('--cluster',
                        type=str, choices=['kmeans', 'rand'], default='kmeans',
                        help="The cluster method")
    parser.add_argument('--train_cluster_number', '--train_size', dest="train_size",
                        type=int, default=10,
                        help="The size of training dataset")
    parser.add_argument('--test_split_rate',
                        type=float, default=0.25,
                        help="The split rate for the real datasets")
    parser.add_argument('--dataset', '-d',
                        type=str, choices=['syn', 'breast_cancer', 'digits'], default='syn',
                        help="The dataset being used")
    parser.add_argument('--n_jobs', '-n',
                        type=int, default=-1,
                        help="Number of workers")
    parser.add_argument('--plot_rate',
                        type=float, default=0.75,
                        help="The plot rate of x axis")
    parser.add_argument('--output', '-o',
                        type=str, default='data_removing_task',
                        help="The output prefix")

    data_parser = parser.add_argument_group('Datasets')
    data_parser.add_argument('--data_home',
                             type=str, default=os.path.join(os.environ['PWD'], 'scikit_learn_data'),
                             help="The data home to be stored")
    data_parser.add_argument('--nodownload', dest='download',
                             action='store_false',
                             help="Whether download the data or not")

    syn_parser = parser.add_argument_group('Synthetic Data')
    syn_parser.add_argument('--train_cluster_size',
                            type=int, default=1,
                            help="The average number of instances in each train cluster")
    syn_parser.add_argument('--test_size',
                            type=int, default=500,
                            help="The size of test dataset")
    syn_parser.add_argument('--dim',
                            type=int, default=6,
                            help="The dimension of the synthetic data")
    syn_parser.add_argument('--difficulty',
                            type=int, default=2,
                            help="The difficulty of the data")
    syn_parser.add_argument('--tol',
                            type=float, default=0.03,
                            help="The tolerance")
    syn_parser.add_argument('--target_accuracy',
                            type=float, default=0.7,
                            help="The target accuracy of the model")
    syn_parser.add_argument('--important_dims',
                            type=int, default=4,
                            help="The imprtant dimension")
    syn_parser.add_argument('--num_class', '-c',
                            type=int, default=2,
                            help="The number of classes")
    syn_parser.add_argument('--param',
                            type=float, default=1.0,
                            help="The param for synthetic data")

    run_parser = parser.add_argument_group('Running Parameters')
    run_parser.add_argument("--tempe",
                            type=float, default=0.1,
                            help="The temperature for game EBM")
    run_parser.add_argument('--test_metric',
                            type=str, choices=['auc', 'accuracy', 'f1', 'xe'], default='accuracy',
                            help="The test metric for base model")
    run_parser.add_argument('--num_test',
                            type=int, default=0,
                            help="The number of tests")
    run_parser.add_argument('--num_train',
                            type=int, default=0,
                            help="The number of trains")
    run_parser.add_argument('--tmp_dir', '-t',
                            type=str, default="./temp",
                            help="The temp dir for data")
    run_parser.add_argument('--reset_tmp', '-r',
                            action='store_true',
                            help="Whether reset the temp dir")

    args = parser.parse_args()

    problem, model = args.problem, args.model
    hidden_units = []  # Empty list in the case of logistic regression.
    if 'all' in args.val_criterion:
        args.val_criterion.extend(['vi', 'shapley', 'banzhaf'])
    # args.shap_metric.append('random')

    clf = get_model(model, solver='liblinear', hidden_units=tuple(hidden_units))
    cluster_number = args.train_size
    print(f'Loading {args.dataset} data ...')

    if args.dataset == 'syn':
        # Synthetic data
        d, difficulty = args.dim, args.difficulty
        # num_classes = args.num_class
        # tol = args.tol
        target_accuracy = args.target_accuracy
        important_dims = args.important_dims
        train_size = args.train_size * args.train_cluster_size
        test_size = args.test_size
        _param = args.param

        X_best, y_best, y_true_best, best_acc = None, None, None, 0.0
        for _ in range(100):
            X_raw = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d),
                                                  size=train_size + test_size)
            _, y_raw, y_true, _ = label_generator(problem, X_raw, param=_param, difficulty=difficulty,
                                                  important=important_dims)

            clf.fit(X_raw[:train_size], y_raw[:train_size])
            test_acc = clf.score(X_raw[train_size:], y_raw[train_size:])

            if test_acc > best_acc:
                best_acc = test_acc
                X_best = X_raw.copy()
                y_best = y_raw.copy()
                y_true_best = y_true.copy()
            if test_acc > target_accuracy:
                break
            _param *= 1.1
        test_acc = best_acc
        X_raw = X_best
        y_raw = y_best
        y_true = y_true_best
        print(f'Performance using the whole training set:  {test_acc:.2f}')
        print(f'The shape of Y is {*y_true.shape,}')

        X, y = X_raw[:train_size], y_raw[:train_size]
        X_test, y_test = X_raw[train_size:], y_raw[train_size:]
        num_test = args.num_test

    elif args.dataset == 'breast_cancer':
        data = load_breast_cancer()
        num_test = args.num_test
        X_raw, y_raw, d = data.data, data.target, data.data.shape[1]
        X, X_test, y, y_test = train_test_split(X_raw, y_raw, test_size=args.test_split_rate,
                                                stratify=y_raw if problem == 'classification' else None)
    elif args.dataset == 'digits':

        data = load_digits()
        num_test = args.num_test
        X_raw, y_raw, d = data.data, data.target, data.data.shape[1]
        X, X_test, y, y_test = train_test_split(X_raw, y_raw, test_size=args.test_split_rate,
                                                stratify=y_raw if problem == 'classification' else None)
    else:
        raise NotImplementedError("The dataset type is not supported!")

    clf.fit(X, y)
    print(f"The {args.test_metric} of the dataset is {clf.score(X_test, y_test):.2f}")

    # running
    directory = args.tmp_dir
    if cluster_number < len(X) and args.train_cluster_size >= 1:
        sources = clustering(args.cluster, X, cluster_number, n_jobs=args.n_jobs)
        print(f'The size of sources is {len(sources):d} with each {*map(lambda _: len(_[1]), sources.items()),}')
    else:
        sources = None

    if args.reset_tmp:
        if os.path.isdir(directory):
            try:
                shutil.rmtree(directory)
            except OSError as e:
                print(f'Warning: failed to reset temp directory {directory} with error {e.strerror}.')
        else:
            print(f'Warning: {directory} does not exist or is not a directory. Ignoring the reset temp directory...')

    print(f'The shape of Train X = {*X.shape,}')

    dshap = DShap(X, y, X_test, y_test, args.num_train, num_test, sources=sources, model_family=model,
                  metric=args.test_metric, directory=directory, seed=0)

    dshap.compute_true_vals(metric=args.test_metric, tempe=args.tempe, n_jobs=args.n_jobs or 1)

    for name, res in dshap.results.items():
        print(f'The values of {name} = {*res,}')

    # Plot the results
    dshap.performance_plots(dshap.results,
                            name=args.output,
                            tempe=args.tempe,
                            num_plot_markers=20,
                            val_criterion=args.val_criterion,
                            percent=args.plot_rate)


if __name__ == "__main__":
    main()
