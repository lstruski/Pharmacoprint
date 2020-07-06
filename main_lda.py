import argparse
from pathlib import Path

import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from data import read_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_act', required=True, help='Path to data.')
    # parser.add_argument('--data_in', default=None, help='Path to data.')
    parser.add_argument('--class_weight', action='store_true', default=None, help='Use balance weight')
    parser.add_argument('--scale_zinc', type=int, default=1)

    args = parser.parse_args()

    args.data_act = './on_bits/A2A_act_onbits'
    args.data_in = './on_bits/A2A_in_onbits'

    if args.class_weight:
        args.class_weight = 'balanced'
    else:
        args.class_weight = None
    print(args)

    if args.data_in is None:
        data = read_data(args.data_act)
        labels = np.ones(data.shape[0])

        np.random.seed(1234)
        zinc = read_data('{}/random_zinc_onbits'.format(Path(args.data_act).parent))
        idx = np.random.choice(zinc.shape[0], size=args.scale_zinc * data.shape[0])
        data = np.concatenate((data, zinc[idx]), axis=0)
        labels = np.concatenate((labels, -np.ones(data.shape[0] - labels.size)))
    else:
        data = read_data(args.data_act)
        labels = np.ones(data.shape[0])
        data = np.concatenate((data, read_data(args.data_in)), axis=0)
        labels = np.concatenate((labels, -np.ones(data.shape[0] - labels.size)))

    print('Data shape before delete columns with the same element in column:', data.shape)
    idx = np.concatenate((np.where((data == 0).all(axis=0))[0], np.where((data == 1).all(axis=0))[0]))

    # test
    for i in idx:
        tmp = data[:, idx[0]].sum()
        assert tmp == 0 or tmp == data.shape[0], "Column '{:d}' does not have the same value!".format(i)

    idx = np.setdiff1d(np.arange(data.shape[1]), idx)
    data = data[:, idx]
    print('Data shape:', data.shape)

    n_splits = 3
    lda = LinearDiscriminantAnalysis(n_components=100)
    scoring = {'acc': make_scorer(accuracy_score)}
    parameters_svc = {'kernel': ('linear', 'rbf'), 'C': [0.01, 0.1, 1, 10, 100]}
    svc = svm.SVC(gamma="scale", class_weight=args.class_weight)
    parameters_lreg = {'C': [0.01, 0.1, 1, 10, 100]}
    lreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', class_weight=args.class_weight,
                              n_jobs=-1)
    lda_acc = np.zeros(n_splits)
    svm_train_acc = np.zeros((n_splits, len(parameters_svc['kernel']) * len(parameters_svc['C'])))
    svm_test_acc = np.zeros((n_splits, len(parameters_svc['kernel']) * len(parameters_svc['C'])))
    lreg_train_acc = np.zeros((n_splits, len(parameters_svc['C'])))
    lreg_test_acc = np.zeros((n_splits, len(parameters_svc['C'])))

    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)
    for i, (train_index, test_index) in enumerate(sss.split(data, labels)):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        X_train_new = lda.fit_transform(X_train, y_train)
        X_test_new = lda.transform(X_test)
        pred = lda.predict(X_test)
        lda_acc[i] = accuracy_score(y_test, pred)

        X = np.concatenate((X_train_new, X_test_new), axis=0)
        y = np.concatenate((y_train, y_test))
        svc.fit(X, y)

        cv = [(np.arange(y_train.size), np.arange(y_train.size, y.size))]
        clf = GridSearchCV(svc, parameters_svc, cv=cv, n_jobs=-1, scoring=scoring, refit='acc', return_train_score=True)
        clf.fit(X, y)
        svm_train_acc[i] = clf.cv_results_['mean_train_acc']
        svm_test_acc[i] = clf.cv_results_['mean_test_acc'] 

        clf = GridSearchCV(lreg, parameters_lreg, cv=cv, n_jobs=-1, scoring=scoring, refit='acc', return_train_score=True)
        clf.fit(X, y)
        lreg_train_acc[i] = clf.cv_results_['mean_train_acc']
        lreg_test_acc[i] = clf.cv_results_['mean_test_acc']

    print('LDA score accuracy:', np.mean(lda_acc))
    id = np.argmax(np.mean(svm_train_acc, axis=0))
    print('SVM score accuracy:', np.mean(svm_test_acc, axis=0)[id])
    id = np.argmax(np.mean(lreg_train_acc, axis=0))
    print('LDA score accuracy:', np.mean(lreg_test_acc, axis=0)[id])
