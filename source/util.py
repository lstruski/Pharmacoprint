def save_results(filename, mode, modelname, gridsearch, head=False, **kwargs):
    with open(filename, mode) as f:
        f.write(f'{modelname};')
        if head:
            f.write('params')
            for name_split in ['train', 'test']:
                for name_sc in ['roc_auc', 'bal', 'acc', 'mcc', 'recall']:
                    for name_ in ['mean', 'std']:
                        f.write(f';{name_}_{name_split}_{name_sc}')
            for name_split in ['fit', 'score']:
                for name_ in ['mean', 'std']:
                    f.write(f';{name_}_{name_split}_time')
            for key, _ in kwargs.items():
                f.write(f';{key}')
        else:
            f.write(','.join(
                f'{key}: {val}' for key, val in gridsearch.cv_results_['params'][gridsearch.best_index_].items()))
            for name_split in ['train', 'test']:
                for name_sc in ['roc_auc', 'bal', 'acc', 'mcc', 'recall']:
                    for name_ in ['mean', 'std']:
                        f.write(
                            f';{gridsearch.cv_results_[f"{name_}_{name_split}_{name_sc}"][gridsearch.best_index_]:.5f}')
            for name_split in ['fit', 'score']:
                for name_ in ['mean', 'std']:
                    f.write(f';{gridsearch.cv_results_[f"{name_}_{name_split}_time"][gridsearch.best_index_]:.5f}')
            for _, val in kwargs.items():
                f.write(f';{val:5f}')
        f.write('\n')
