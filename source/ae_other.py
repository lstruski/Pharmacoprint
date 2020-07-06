import argparse
# from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from scipy.sparse import load_npz
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm


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


def train_step(model_AE, criterion_AE, optimizer, data_train, target, device, writer_tensorboard, n_epoch, batch_size):
    model_AE.train()

    # create batch data
    active_id = np.where(target == 1)[0]
    inactive_id = np.setdiff1d(np.arange(target.size), active_id)
    choice_samples = active_id.size - inactive_id.size
    if choice_samples > 0:
        inactive_id = np.concatenate((inactive_id, np.random.choice(inactive_id, size=choice_samples, replace=True)))
    else:
        active_id = np.concatenate((active_id, np.random.choice(active_id, size=-choice_samples, replace=True)))

    np.random.shuffle(active_id)
    np.random.shuffle(inactive_id)
    n_batch, rest = divmod(active_id.size, batch_size)
    n_batch = n_batch + 1 if rest else n_batch

    train_tqdm = tqdm(range(n_batch), desc="Train loss", leave=False)
    for i in train_tqdm:
        batch_idx = active_id[i * batch_size:(i + 1) * batch_size]
        batch_idx = np.concatenate((batch_idx, inactive_id[i * batch_size:(i + 1) * batch_size]), axis=0)

        idx = np.random.permutation(np.arange(batch_idx.size))
        batch_idx = batch_idx[idx]

        batch = torch.from_numpy(data_train[batch_idx]).float().to(device)
        # ===================forward=====================
        z, output_ae = model_AE(batch)

        loss = criterion_AE(output_ae, batch)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # ===================log========================
        train_tqdm.set_description("Train loss AE: {:.5f}".format(loss.item()))
        writer_tensorboard.add_scalar('pre_train/trn_ae', loss, n_epoch * n_batch + i)
    return loss.item()


def test_step(model_AE, criterion_AE, data_test, target, device, writer_tensorboard, n_epoch, batch_size):
    model_AE.eval()
    test_loss = 0

    # create batch data
    active_id = np.where(target == 1)[0]
    inactive_id = np.setdiff1d(np.arange(target.size), active_id)
    choice_samples = active_id.size - inactive_id.size
    if choice_samples > 0:
        inactive_id = np.concatenate((inactive_id, np.random.choice(inactive_id, size=choice_samples, replace=True)))
    else:
        active_id = np.concatenate((active_id, np.random.choice(active_id, size=-choice_samples, replace=True)))

    np.random.shuffle(active_id)
    np.random.shuffle(inactive_id)
    n_batch, rest = divmod(active_id.size, batch_size)
    n_batch = n_batch + 1 if rest else n_batch

    with torch.no_grad():
        test_tqdm = tqdm(range(n_batch), desc="Test loss", leave=False)
        for i in test_tqdm:
            batch_idx = active_id[i * batch_size:(i + 1) * batch_size]
            batch_idx = np.concatenate((batch_idx, inactive_id[i * batch_size:(i + 1) * batch_size]), axis=0)

            idx = np.random.permutation(np.arange(batch_idx.size))
            batch_idx = batch_idx[idx]

            batch = torch.from_numpy(data_test[batch_idx]).float().to(device)
            # ===================forward=====================
            z, output_ae = model_AE(batch)

            loss = criterion_AE(output_ae, batch)
            test_loss += loss.item()  # sum up batch loss
    # ===================log========================
    test_loss /= n_batch
    test_tqdm.set_description("Test loss: {:.5f}".format(test_loss))
    writer_tensorboard.add_scalar('pre_train/tst_ae', test_loss, n_epoch)
    return test_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data', help='Name/path of target')
    parser.add_argument('--name', default=['5HT2A'], nargs='+', help='Name of target')

    parser.add_argument('--class_weight', action='store_true', default=None, help='Use balance weight')

    parser.add_argument('--pretrain_epochs', type=int, default=100, help="Number of epochs to pretrain model AE")
    parser.add_argument('--dims_layers_ae', type=int, nargs='+', default=[500, 100, 10],
                        help="Dimensional of layers in AE")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--use_dropout', action='store_true', help="Use dropout")

    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1)')

    parser.add_argument('--save_dir', default='./outputs', help='Path to dictionary where will be save results.')
    parser.add_argument('--ae', default=None, help='Path to saved model.')
    parser.add_argument('--test', action='store_true', help='Test model')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    id_file_target = 0
    # current_date = datetime.now()
    # current_date = current_date.strftime('%d%b_%H%M%S')
    save_dir = f'{args.save_dir}/tensorboard/{args.name[id_file_target]}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if 'pca' in PurePosixPath(args.data_dir).stem:
        data = np.load(f'{args.data_dir}/{args.name[id_file_target]}.npz')['data']
    else:
        data = load_npz(f'{args.data_dir}/{args.name[id_file_target]}.npz').todense()
    labels = np.load(f'{args.data_dir}/../label/{args.name[id_file_target]}.npz')['lab']

    args.dims_layers_ae = [data.shape[1]] + args.dims_layers_ae

    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(data, labels):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        del train_index, test_index
        break

    model_ae = AutoEncoder(args.dims_layers_ae, args.use_dropout).to(device)
    if args.ae is not None:
        if device.type == "cpu":
            model_ae.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
        else:
            model_ae.load_state_dict(torch.load(args.model))
        model_ae.eval()

    criterion_ae = nn.MSELoss()
    optimizer = torch.optim.Adam(model_ae.parameters(), lr=args.lr, weight_decay=1e-5)

    writer = SummaryWriter(logdir=save_dir)
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
    clf = GridSearchCV(svc, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc', return_train_score=True)
    clf.fit(low_data, labels)

    save_file = f'{args.save_dir}/{args.name[id_file_target]}.txt'

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
    clf = GridSearchCV(svc, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc', return_train_score=True)
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
    clf = GridSearchCV(lreg, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc', return_train_score=True,
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
