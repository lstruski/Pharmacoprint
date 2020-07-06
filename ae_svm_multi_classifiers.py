import argparse
from functools import reduce
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm

from data import read_data


class AutoEncoder(nn.Module):
    def __init__(self, dims_layers, dropout=False):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential()
        for i in range(1, len(dims_layers)):
            self.encoder.add_module('enc_dense{:d}'.format(i), nn.Linear(dims_layers[i - 1], dims_layers[i]))
            self.encoder.add_module('enc_relu{:d}'.format(i), nn.ReLU(True))
            if dropout:
                self.encoder.add_module('enc_dropout{:d}'.format(i), nn.Dropout(0.25))

        self.decoder = nn.Sequential()
        size = len(dims_layers) - 1
        for i in range(size - 1, 0, -1):
            self.decoder.add_module('dec_dense{:d}'.format(size - i), nn.Linear(dims_layers[i + 1], dims_layers[i]))
            self.decoder.add_module('dec_relu{:d}'.format(size - i), nn.ReLU(True))
            if dropout:
                self.decoder.add_module('dec_dropout{:d}'.format(size - i), nn.Dropout(0.25))
        self.decoder.add_module('dec_dense{:d}'.format(size), nn.Linear(dims_layers[1], dims_layers[0]))
        self.decoder.add_module('sigmoid', nn.Sigmoid())

    def forward(self, x):
        output_encoder = self.encoder(x)
        output_decoder = self.decoder(output_encoder)
        return output_encoder, output_decoder

    def forward_encoder(self, x):
        return self.encoder(x)


class GetBatch:
    def __init__(self, target, batch_size):
        self.target = target
        self.batch_size = batch_size
        active_id = np.where(self.target == 1)[0]
        inactive_id = np.setdiff1d(np.arange(self.target.size), active_id)
        choice_samples = active_id.size - inactive_id.size
        if choice_samples > 0:
            inactive_id = np.concatenate(
                (inactive_id, np.random.choice(inactive_id, size=choice_samples, replace=True)))
        else:
            active_id = np.concatenate((active_id, np.random.choice(active_id, size=-choice_samples, replace=True)))

        np.random.shuffle(active_id)
        np.random.shuffle(inactive_id)

        self.len, rest = divmod(active_id.size, self.batch_size)
        self.len = self.len + 1 if rest > 0 else self.len
        self.active_id = active_id
        self.inactive_id = inactive_id

    def __call__(self):
        for i in range(self.len):
            yield np.concatenate((self.active_id[i * self.batch_size:(i + 1) * self.batch_size],
                                  self.inactive_id[i * self.batch_size:(i + 1) * self.batch_size]), axis=0)

    def __len__(self):
        return self.len


def train_step(model_AE, criterion_AE, optimizer, data_train, target, device, writer_tensorboard, n_epoch, batch_size):
    model_AE.train()
    batch_iters = []
    max_num_batch = 0
    for i in range(len(data_train)):
        gen = GetBatch(target[i], batch_size)
        batch_iters.append(gen())
        if max_num_batch < len(gen):
            max_num_batch = len(gen)

    train_tqdm = tqdm(range(max_num_batch), desc="Train loss", leave=False)
    for b in train_tqdm:
        loss = torch.tensor([0]).float().to(device)
        for i in range(len(data_train)):
            try:
                batch_idx = next(batch_iters[i])
            except StopIteration:
                batch_iters[i] = GetBatch(target[i], batch_size)()
                batch_idx = next(batch_iters[i])

            batch = torch.from_numpy(data_train[i][batch_idx]).float().to(device)
            # ===================forward=====================
            z, output_ae = model_AE(batch)
            loss = loss + criterion_AE(output_ae, batch)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
        train_tqdm.set_description("Train loss AE: {:.5f}".format(loss.item() / len(data_train)))
        writer_tensorboard.add_scalar('pre_train/trn_ae', loss / len(data_train), n_epoch * max_num_batch + b)
    return loss.item()


def test_step(model_AE, criterion_AE, data_test, target, device, writer_tensorboard, n_epoch, batch_size):
    model_AE.eval()
    batch_iters = []
    num_batch = np.zeros(len(data_test), dtype=int)
    for i in range(len(data_test)):
        gen = GetBatch(target[i], batch_size)
        batch_iters.append(gen())
        num_batch[i] = len(gen)
    max_num_batch = num_batch.max()
    test_loss = 0

    with torch.no_grad():
        for _ in tqdm(range(max_num_batch), leave=False):
            for i in range(len(data_test)):
                try:
                    batch_idx = next(batch_iters[i])
                except StopIteration:
                    continue
                    # batch_iters[i] = GetBatch(target[i], batch_size)()
                    # batch_idx = next(batch_iters[i])

                batch = torch.from_numpy(data_test[i][batch_idx]).float().to(device)
                # ===================forward=====================
                z, output_ae = model_AE(batch)
                loss = criterion_AE(output_ae, batch)
                test_loss += loss.item()
    # ===================log========================
    test_loss /= num_batch.sum()
    writer_tensorboard.add_scalar('pre_train/tst_ae', test_loss, n_epoch)
    return test_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_act', nargs='+', required=True, help='Path to data.')
    parser.add_argument('--data_in', nargs='+', default=None, help='Path to data.')
    parser.add_argument('--data_zinc', default=None, help='Path to data.')
    parser.add_argument('--scale_zinc', type=int, default=1)
    parser.add_argument('--class_weight', action='store_true', default=None, help='Use balance weight')

    parser.add_argument('--pretrain_epochs', type=int, default=100, help="Number of epochs to pretrain model AE")
    parser.add_argument('--dims_layers_ae', type=int, nargs='+', default=[500, 100, 10],
                        help="Dimensional of layers in AE")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--use_dropout', action='store_true', help="Use dropout")

    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1)')

    parser.add_argument('--dir_save', default='./outputs', help='Path to dictionary where will be save results.')
    parser.add_argument('--ae', default=None, help='Path to saved model.')
    parser.add_argument('--test', action='store_true', help='Test model')
    args = parser.parse_args()

    # current_date = datetime.now()
    # current_date = current_date.strftime('%d%b_%H%M%S')
    dir_save = '{}/tensorboard'.format(args.dir_save)
    Path(dir_save).mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # dataset
    if args.data_zinc is not None:
        data = read_data(args.data_act)
        labels = np.ones(data.shape[0])

        np.random.seed(args.seed)
        zinc = read_data(args.data_zinc)
        idx = np.random.choice(zinc.shape[0], size=args.scale_zinc * data.shape[0])
        data = np.concatenate((data, zinc[idx]), axis=0)
        labels = np.concatenate((labels, -np.ones(data.shape[0] - labels.size)))
    elif args.data_in is not None:
        datasets = []
        labels = []
        for i in range(len(args.data_act)):
            data = read_data(args.data_act[i])
            label = np.ones(data.shape[0])
            data = np.concatenate((data, read_data(args.data_in[i])), axis=0)
            label = np.concatenate((label, np.zeros(data.shape[0] - label.size)))
            datasets.append(data)
            labels.append(label)
        data = np.concatenate(datasets, axis=0)
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

    args.dims_layers_ae = [data.shape[1]] + args.dims_layers_ae

    sizes = [i.size for i in labels]
    sizes = [reduce((lambda v1, v2: v1 + v2), sizes[:i]) for i in range(1, len(sizes))]
    data = np.split(data, sizes)
    num_datasets = len(data)

    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i in range(num_datasets):
        for train_index, test_index in sss.split(data[i], labels[i]):
            # print("TRAIN:", train_index, "TEST:", test_index)
            x_train.append(data[i][train_index])
            x_test.append(data[i][test_index])
            y_train.append(labels[i][train_index])
            y_test.append(labels[i][test_index])
            break
    del train_index, test_index, data

    model_ae = AutoEncoder(args.dims_layers_ae, args.use_dropout).to(device)
    if args.ae is not None:
        if device.type == "cpu":
            model_ae.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
        else:
            model_ae.load_state_dict(torch.load(args.model))
        model_ae.eval()

    criterion_ae = nn.MSELoss()
    optimizer = torch.optim.Adam(model_ae.parameters(), lr=args.lr, weight_decay=1e-5)

    writer = SummaryWriter(logdir=dir_save)
    epoch_tqdm = tqdm(range(args.pretrain_epochs), desc="Epoch pre-train loss")
    for epoch in epoch_tqdm:
        loss_train = train_step(model_ae, criterion_ae, optimizer,
                                x_train, y_train, device, writer, epoch, args.batch_size)
        loss_test = test_step(model_ae, criterion_ae, x_test, y_test,
                              device, writer, epoch, args.batch_size)
        epoch_tqdm.set_description("Epoch pre-train loss: {:.5f}, test loss: {:.5f}".format(loss_train, loss_test))

    print('===================================================')
    print('Transforming data to lower dimensional')
    model_ae.eval()

    for idx_cl in range(len(x_train)):
        data = np.concatenate((x_train[idx_cl], x_test[idx_cl]), axis=0)
        labels = np.concatenate((y_train[idx_cl], y_test[idx_cl]), axis=0)

        low_data = np.empty((data.shape[0], args.dims_layers_ae[-1]))

        n_batch, rest = divmod(data.shape[0], args.batch_size)
        n_batch = n_batch + 1 if rest else n_batch

        with torch.no_grad():
            test_tqdm = tqdm(range(n_batch), desc="Transform data", leave=False)
            for i in test_tqdm:
                batch = torch.from_numpy(data[i * args.batch_size:(i + 1) * args.batch_size, :]).float().to(device)
                # ===================forward=====================
                z, _ = model_ae(batch)
                low_data[i * args.batch_size:(i + 1) * args.batch_size, :] = z.detach().cpu().numpy()
        print('Data shape after transformation: {}'.format(low_data.shape))
        print('===================================================')

        if args.class_weight:
            args.class_weight = 'balanced'
        else:
            args.class_weight = None

        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
        scoring = {'acc': make_scorer(accuracy_score), 'roc_auc': make_scorer(roc_auc_score),
                   'mcc': make_scorer(matthews_corrcoef)}

        # Linear SVM
        parameters = {'C': [0.01, 0.1, 1, 10, 100]}
        svc = svm.LinearSVC(class_weight=args.class_weight, random_state=args.seed)
        clf = GridSearchCV(svc, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc',
                           return_train_score=True)
        clf.fit(low_data, labels)

        save_file = '{}/{}.txt'.format(args.dir_save, PurePosixPath(args.data_act[idx_cl]).stem.replace('_act', ''))

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
        svc = svm.SVC(gamma="scale", class_weight=args.class_weight, random_state=args.seed)
        clf = GridSearchCV(svc, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc',
                           return_train_score=True)
        clf.fit(low_data, labels)
        with open(save_file, 'a') as f:
            f.write('{:.4f} {:.5f} | ACC\n'.format(clf.cv_results_['mean_test_acc'][clf.best_index_],
                                                   clf.cv_results_['std_test_acc'][clf.best_index_]))
            f.write('{:.4f} {:.5f} | ROC_AUC\n'.format(clf.cv_results_['mean_test_roc_auc'][clf.best_index_],
                                                       clf.cv_results_['std_test_roc_auc'][clf.best_index_]))
            f.write('{:.4f} {:.5f} | MCC\n'.format(clf.cv_results_['mean_test_mcc'][clf.best_index_],
                                                   clf.cv_results_['std_test_mcc'][clf.best_index_]))
            f.write('\nBest_score for SVC RBF {:.4f} roc_auc\n'.format(clf.best_score_))
            f.write('----------------------------------------------------\n')

        lreg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', class_weight=args.class_weight,
                                  n_jobs=-1)
        parameters = {'C': [0.01, 0.1, 1, 10, 100]}
        clf = GridSearchCV(lreg, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc',
                           return_train_score=True,
                           error_score='raise')
        clf.fit(low_data, labels)

        with open(save_file, 'a') as f:
            f.write('{:.4f} {:.5f} | ACC\n'.format(clf.cv_results_['mean_test_acc'][clf.best_index_],
                                                   clf.cv_results_['std_test_acc'][clf.best_index_]))
            f.write('{:.4f} {:.5f} | ROC_AUC\n'.format(clf.cv_results_['mean_test_roc_auc'][clf.best_index_],
                                                       clf.cv_results_['std_test_roc_auc'][clf.best_index_]))
            f.write('{:.4f} {:.5f} | MCC\n'.format(clf.cv_results_['mean_test_mcc'][clf.best_index_],
                                                   clf.cv_results_['std_test_mcc'][clf.best_index_]))
            f.write('\nBest_score for LogisticRegression {:.4f} roc_auc\n'.format(clf.best_score_))
            f.write('----------------------------------------------------\n')


if __name__ == '__main__':
    main()
