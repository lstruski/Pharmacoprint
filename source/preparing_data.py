import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, save_npz
from sklearn.decomposition import PCA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../data/inne_fingerprinty', help='Path of target')
    parser.add_argument('--pattern', default='_act_3D_H_')
    parser.add_argument('--dir_save', default='./data', help='Path to dictionary where will be save results.')
    parser.add_argument('--pca', type=float, default=None)
    args = parser.parse_args()

    args.pca = int(args.pca) if args.pca is not None and args.pca > 1 else args.pca

    path_data = f'{args.dir_save}/data'
    path_label = f'{args.dir_save}/label'
    path_all_data = f'{args.dir_save}/all_data'
    path_pca = f'{args.dir_save}/pca_{args.pca}'
    path_all_pca = f'{args.dir_save}/all_data_pca_{args.pca}'

    Path(path_label).mkdir(parents=True, exist_ok=True)
    Path(path_data).mkdir(parents=True, exist_ok=True)
    Path(path_all_data).mkdir(parents=True, exist_ok=True)
    if args.pca is not None:
        Path(path_pca).mkdir(parents=True, exist_ok=True)
        Path(path_all_pca).mkdir(parents=True, exist_ok=True)

    path = Path(args.data_dir)
    print(path)
    files = list(path.glob(f'*{args.pattern}*.csv'))
    # print(files)

    targets = set()
    for f in files:
        targets.update([f.name.split(args.pattern)[0]])
    targets = list(targets)
    # print(sorted(targets))
    # print(len(targets))

    data_all = []
    for t in targets:
        print(f'Target: {t}')
        files = list(path.glob(f'{t}{args.pattern}*.csv'))
        files = sorted(files, key=lambda p: p.name)

        data_act = []
        data_in = []
        for file in files:
            data_act.append(pd.read_csv(file, delimiter=',', header=0).iloc[:, 1:].to_numpy())
            data_in.append(pd.read_csv(path.joinpath(file.name.replace("act", "in")),
                                       delimiter=',', header=0).iloc[:, 1:].to_numpy())
            assert np.sum(np.isnan(data_act[-1])) == 0, f'NAN is in "{file.name}", {np.where(np.isnan(data_act[-1]))}'
            assert np.sum(np.isnan(data_in[-1])) == 0, f'NAN is in "{file.name.replace("act", "in")}", ' \
                                                       f'{np.where(np.isnan(data_in[-1]))}'
        data_act = np.concatenate(data_act, axis=1)
        data_in = np.concatenate(data_in, axis=1)
        assert data_act.shape[1] == data_in.shape[1]
        data = np.concatenate((data_act, data_in), axis=0)
        labels = np.ones(data.shape[0])
        labels[data_act.shape[0]:] = 0
        del data_act, data_in
        data_all.append(data)

        print('Data shape before delete columns with the same element in column:', data.shape)
        idx = np.concatenate((np.where((data == 0).all(axis=0))[0], np.where((data == 1).all(axis=0))[0]))

        # test
        for i in idx:
            tmp = data[:, idx[0]].sum()
            assert tmp == 0 or tmp == data.shape[0], f"Column '{i:d}' does not have the same value!"

        idx = np.setdiff1d(np.arange(data.shape[1]), idx)
        data = data[:, idx]
        print('Data shape:', data.shape)

        save_npz(f'{path_data}/{t}.npz', csc_matrix(data))
        np.savez(f'{path_label}/{t}.npz', lab=labels)

        if args.pca is not None:
            dim_pca = data.shape[1]
            data = PCA(n_components=args.pca).fit_transform(data)
            print(f'DIM before PCA {dim_pca:d}, after {data.shape[1]:d}')
            np.savez(f'{path_pca}/{t}.npz', data=data)
        print('-----------------------------------')

    n_rows = list(map(lambda x: x.shape[0], data_all))
    data_all = np.concatenate(data_all, axis=0)
    print('Data all shape before delete columns with the same element in column:', data_all.shape)
    idx = np.concatenate((np.where((data_all == 0).all(axis=0))[0], np.where((data_all == 1).all(axis=0))[0]))

    # test
    for i in idx:
        tmp = data_all[:, idx[0]].sum()
        assert tmp == 0 or tmp == data_all.shape[0], f"Column '{i:d}' does not have the same value!"

    idx = np.setdiff1d(np.arange(data_all.shape[1]), idx)

    data_all = data_all[:, idx]
    print('Data shape:', data_all.shape)

    j = 0
    for i in range(len(targets)):
        save_npz(f'{path_all_data}/{targets[i]}.npz', csc_matrix(data_all[j:j + n_rows[i]]))
        j += n_rows[i]
    if args.pca is not None:
        dim_pca = data_all.shape[1]
        data_all = PCA(n_components=args.pca).fit_transform(data_all)
        print(f'DIM before PCA {dim_pca:d}, after {data_all.shape[1]:d}')
        j = 0
        for i in range(len(targets)):
            np.savez(f'{path_all_pca}/{targets[i]}.npz', data=data_all[j:j + n_rows[i]])
            j += n_rows[i]


if __name__ == '__main__':
    main()
