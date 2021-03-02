import argparse
import warnings
from os import environ
from pathlib import Path, PurePosixPath

import numpy as np
from sklearn import svm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

from util import save_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=True, help='Name/path of file')
    parser.add_argument('--savefile', type=str, default='./output.txt', help='Path to file where will be save results')
    parser.add_argument('--class_weight', action='store_true', default=None, help='Use balance weight')
    parser.add_argument('--seed', default=1234, help='Number of seed')
    args = parser.parse_args()
    if args.class_weight:
        args.class_weight = 'balanced'
    else:
        args.class_weight = None
    print(args)
    np.random.seed(args.seed)

    loaded = np.load(args.filename)
    data = loaded['data']
    labels = loaded['label']
    del loaded

    Path(PurePosixPath(args.savefile).parent).mkdir(parents=True, exist_ok=True)

    # Split data
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.1, random_state=args.seed)
    scoring = {'acc': make_scorer(accuracy_score), 'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
               'mcc': make_scorer(matthews_corrcoef), 'bal': make_scorer(balanced_accuracy_score),
               'recall': make_scorer(recall_score)}

    max_iters = 10000
    save_results(args.savefile, 'w', 'model', None, True)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses (n_jobs > 1)

        # Linear SVM
        print("\rLinear SVM         ", end='')
        parameters = {'C': [0.01, 0.1, 1, 10, 100]}
        # svc = svm.LinearSVC(class_weight=args.class_weight, random_state=seed)
        svc = svm.SVC(kernel='linear', class_weight=args.class_weight, random_state=args.seed, probability=True,
                      max_iter=max_iters)
        clf = GridSearchCV(svc, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc',
                           return_train_score=True)
        try:
            clf.fit(data, labels)
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)

        save_results(args.savefile, 'a', 'Linear SVM', clf)

        # RBF SVM
        print("\rRBF SVM             ", end='')
        parameters = {'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 1e-2, 1e-3, 1e-4]}
        svc = svm.SVC(gamma="scale", class_weight=args.class_weight, random_state=args.seed, probability=True,
                      max_iter=max_iters)
        clf = GridSearchCV(svc, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc',
                           return_train_score=True)
        try:
            clf.fit(data, labels)
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
        save_results(args.savefile, 'a', 'RBF SVM', clf)

        # LogisticRegression
        print("\rLogisticRegression  ", end='')
        lreg = LogisticRegression(random_state=args.seed, solver='lbfgs', multi_class='ovr',
                                  class_weight=args.class_weight,
                                  n_jobs=-1, max_iter=max_iters)
        parameters = {'C': [0.01, 0.1, 1, 10, 100]}
        clf = GridSearchCV(lreg, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc',
                           return_train_score=True)
        try:
            clf.fit(data, labels)
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
        save_results(args.savefile, 'a', 'LogisticRegression', clf)
        print()
