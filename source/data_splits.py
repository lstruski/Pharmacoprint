import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=True, help='Name/path of file')
    parser.add_argument('--save_dir', default='./outputs', help='Path to dictionary where will be save results.')
    parser.add_argument('--n_splits', default=3, type=int, help='Number of split of data for cross-validation.')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1)')
    args = parser.parse_args()

    np.random.seed(args.seed)

    loaded = np.load(args.filename)
    data = loaded['data']
    labels = loaded['label']
    del loaded

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    sss = StratifiedShuffleSplit(n_splits=args.n_splits, test_size=0.1, random_state=args.seed)

    for i, (train_index, test_index) in enumerate(sss.split(data, labels)):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        np.savez_compressed(f'{args.save_dir}/{i}', data_train=x_train, data_test=x_test,
                            lab_train=y_train, lab_test=y_test)


if __name__ == '__main__':
    main()
