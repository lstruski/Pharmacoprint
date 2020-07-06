import argparse
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

from data import read_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_act', required=True, help='Path to data.')
    parser.add_argument('--data_in', default=None, help='Path to data.')
    parser.add_argument('--data_zinc', default=None, help='Path to data.')
    parser.add_argument('--class_weight', action='store_true', default=None, help='Use balance weight')
    parser.add_argument('--scale_zinc', type=int, default=9)
    parser.add_argument('--pca', type=float, default=None)
    parser.add_argument('--logdir', type=str, default='./logdir', help='Path to directory where save results')
    parser.add_argument('--dense', action="store_true", help='Use dense data')
    args = parser.parse_args()
    if args.class_weight:
        args.class_weight = 'balanced'
    else:
        args.class_weight = None
    if args.pca is not None and args.pca > 1:
        args.pca = int(args.pca)
    print(args)

    seed = 1234

    if args.data_zinc is not None:
        print('\x1b[1;3;41mUsing ZINC\x1b[0m')
        if args.dense:
            data = pd.read_csv(args.data_act, delimiter=',', header=0).iloc[:, 1:].to_numpy()
            labels = np.ones(data.shape[0])

            np.random.seed(seed)
            zinc = pd.read_csv(args.data_zinc, delimiter=',', header=0).iloc[:, 1:].to_numpy()
        else:
            data = read_data(args.data_act)
            labels = np.ones(data.shape[0])

            np.random.seed(seed)
            zinc = read_data(args.data_zinc)
        idx = np.random.choice(zinc.shape[0], size=args.scale_zinc * data.shape[0])
        data = np.concatenate((data, zinc[idx]), axis=0)
        labels = np.concatenate((labels, -np.ones(data.shape[0] - labels.size)))
    elif args.data_in is not None:
        print('\x1b[1;3;41mUsing INACTIVE\x1b[0m')
        # args.data_act = './on_bits/A2A_act_onbits'
        # args.data_in = './on_bits/A2A_in_onbits'
        
        if args.dense:
            data = pd.read_csv(args.data_act, delimiter=',', header=0).iloc[:, 1:].to_numpy()
            labels = np.ones(data.shape[0])
            data = np.concatenate((data, pd.read_csv(args.data_in, delimiter=',', header=0).iloc[:, 1:].to_numpy()), axis=0)
            labels = np.concatenate((labels, -np.ones(data.shape[0] - labels.size)))
        else:
            data = read_data(args.data_act)
            labels = np.ones(data.shape[0])
            data = np.concatenate((data, read_data(args.data_in)), axis=0)
            labels = np.concatenate((labels, -np.ones(data.shape[0] - labels.size)))
    else:
        raise SystemError('Type path to zinc or inactive file')

    print('Data shape before delete columns with the same element in column:', data.shape)
    idx = np.concatenate((np.where((data == 0).all(axis=0))[0], np.where((data == 1).all(axis=0))[0]))

    # test
    for i in idx:
        tmp = data[:, idx[0]].sum()
        assert tmp == 0 or tmp == data.shape[0], "Column '{:d}' does not have the same value!".format(i)

    idx = np.setdiff1d(np.arange(data.shape[1]), idx)
    data = data[:, idx]
    print('Data shape:', data.shape)

    np.random.seed(seed)

    if args.pca is not None:
        dim_pca = data.shape[1]
        data = PCA(n_components=args.pca).fit_transform(data)
        dim_pca = 'DIM before PCA {:d}, after {:d}'.format(dim_pca, data.shape[1])
        print('Use PCA, dimension after transformation:', data.shape)

    Path(args.logdir).mkdir(parents=True, exist_ok=True)

    # Split data
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=seed)
    scoring = {'acc': make_scorer(accuracy_score), 'roc_auc': make_scorer(roc_auc_score),
               'mcc': make_scorer(matthews_corrcoef)}

    # Linear SVM
    parameters = {'C': [0.01, 0.1, 1, 10, 100]}
    svc = svm.LinearSVC(class_weight=args.class_weight, random_state=seed)
    clf = GridSearchCV(svc, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc', return_train_score=True)
    clf.fit(data, labels)

    save_file = '{}/{}.txt'.format(args.logdir, PurePosixPath(args.data_act).stem.replace('_act', ''))

    with open(save_file, 'w') as f:
        f.write('{:.4f} {:.5f} | ACC\n'.format(clf.cv_results_['mean_test_acc'][clf.best_index_],
                                               clf.cv_results_['std_test_acc'][clf.best_index_]))
        f.write('{:.4f} {:.5f} | ROC_AUC\n'.format(clf.cv_results_['mean_test_roc_auc'][clf.best_index_],
                                                   clf.cv_results_['std_test_roc_auc'][clf.best_index_]))
        f.write('{:.4f} {:.5f} | MCC\n'.format(clf.cv_results_['mean_test_mcc'][clf.best_index_],
                                             clf.cv_results_['std_test_mcc'][clf.best_index_]))
        f.write('\nBest_score for LinearSVC {:.4f} roc_auc\n'.format(clf.best_score_))
        f.write('----------------------------------------------------\n')

    # RBF SVM
    parameters = {'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 1e-2, 1e-3, 1e-4]}
    svc = svm.SVC(gamma="scale", class_weight=args.class_weight, random_state=seed)
    clf = GridSearchCV(svc, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc', return_train_score=True)
    clf.fit(data, labels)
    with open(save_file, 'a') as f:
        f.write('{:.4f} {:.5f} | ACC\n'.format(clf.cv_results_['mean_test_acc'][clf.best_index_],
                                               clf.cv_results_['std_test_acc'][clf.best_index_]))
        f.write('{:.4f} {:.5f} | ROC_AUC\n'.format(clf.cv_results_['mean_test_roc_auc'][clf.best_index_],
                                                   clf.cv_results_['std_test_roc_auc'][clf.best_index_]))
        f.write('{:.4f} {:.5f} | MCC\n'.format(clf.cv_results_['mean_test_mcc'][clf.best_index_],
                                               clf.cv_results_['std_test_mcc'][clf.best_index_]))
        f.write('\nBest_score for SVC RBF {:.4f} roc_auc\n'.format(clf.best_score_))
        f.write('----------------------------------------------------\n')

    lreg = LogisticRegression(random_state=seed, solver='lbfgs', multi_class='ovr', class_weight=args.class_weight,
                              n_jobs=-1)
    parameters = {'C': [0.01, 0.1, 1, 10, 100]}
    clf = GridSearchCV(lreg, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc', return_train_score=False)
    clf.fit(data, labels)
    with open(save_file, 'a') as f:
        f.write('{:.4f} {:.5f} | ACC\n'.format(clf.cv_results_['mean_test_acc'][clf.best_index_],
                                               clf.cv_results_['std_test_acc'][clf.best_index_]))
        f.write('{:.4f} {:.5f} | ROC_AUC\n'.format(clf.cv_results_['mean_test_roc_auc'][clf.best_index_],
                                                   clf.cv_results_['std_test_roc_auc'][clf.best_index_]))
        f.write('{:.4f} {:.5f} | MCC\n'.format(clf.cv_results_['mean_test_mcc'][clf.best_index_],
                                               clf.cv_results_['std_test_mcc'][clf.best_index_]))
        f.write('\nBest_score for LogisticRegression {:.4f} roc_auc\n'.format(clf.best_score_))
        f.write('----------------------------------------------------\n')

        if args.pca is not None:
            f.write('{}\n'.format(dim_pca))
