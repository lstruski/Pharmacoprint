import argparse
from pathlib import Path, PurePosixPath

import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

from scipy.sparse import load_npz


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data', help='Name/path of target')
    # ['5HT2A', '5HT2c', '5HT6', 'D2', 'HIVint', 'HIVprot', 'HIVrev', 'NMDA', 'NOP', 'NPC1', 'catB', 'catL', 'delta', 'kappa', 'mi']
    parser.add_argument('--name', default=['5HT2A'], nargs='+', help='Name of target')

    parser.add_argument('--class_weight', action='store_true', default=None, help='Use balance weight')
    parser.add_argument('--logdir', type=str, default='./logdir', help='Path to directory where save results')
    args = parser.parse_args()
    if args.class_weight:
        args.class_weight = 'balanced'
    else:
        args.class_weight = None
    print(args)

    seed = 1234
    np.random.seed(seed)

    id_file_target = 0
    if 'pca' in PurePosixPath(args.data_dir).stem:
        data = np.load(f'{args.data_dir}/{args.name[id_file_target]}.npz')['data']
    else:
        data = load_npz(f'{args.data_dir}/{args.name[id_file_target]}.npz').todense()
    labels = np.load(f'{args.data_dir}/../label/{args.name[id_file_target]}.npz')['lab']

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

    save_file = f'{args.logdir}/{args.name[id_file_target]}.txt'

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
        f.write(f'\nBest_score for SVC RBF {clf.best_score_:.4f} roc_auc\n')
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
        f.write(f'\nBest_score for LogisticRegression {clf.best_score_:.4f} roc_auc\n')
        f.write('----------------------------------------------------\n')
