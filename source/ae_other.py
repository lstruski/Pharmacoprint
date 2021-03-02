import argparse
import time
import warnings
from os import environ
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from sklearn import svm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import AutoEncoder
from util import save_results


def train_step(model_AE, criterion_AE, optimizer, scheduler, data_train, target, device, writer_tensorboard, n_epoch,
               batch_size):
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

    time_work = 0
    train_tqdm = tqdm(range(n_batch), desc="Train loss", leave=False)
    for i in train_tqdm:
        start_time = time.time()
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
        if scheduler is not None:
            scheduler.step()

        end_time = time.time()
        time_work += end_time - start_time
        # ===================log========================
        train_tqdm.set_description("Train loss AE: {:.5f}".format(loss.item()))
        writer_tensorboard.add_scalar('pre_train/trn_ae', loss, n_epoch * n_batch + i)
    return loss.item(), time_work


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
    parser.add_argument('--filename', required=True, help='Name/path of file')
    parser.add_argument('--savefile', type=str, default='./output.txt', help='Path to file where will be save results')
    parser.add_argument('--class_weight', action='store_true', default=None, help='Use balance weight')
    parser.add_argument('--seed', default=1234, help='Number of seed')

    parser.add_argument('--pretrain_epochs', type=int, default=100, help="Number of epochs to pretrain model AE")
    parser.add_argument('--dims_layers_ae', type=int, nargs='+', default=[500, 100, 10],
                        help="Dimensional of layers in AE")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--use_dropout', action='store_true', help="Use dropout")
    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--earlyStopping', type=int, default=None, help='Number of epochs to early stopping')
    parser.add_argument('--use_scheduler', action='store_true')
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Device: {device.type}')

    loaded = np.load(args.filename)
    data = loaded['data']
    labels = loaded['label']
    del loaded

    name_target = PurePosixPath(args.savefile).stem
    save_dir = f'{PurePosixPath(args.savefile).parent}/tensorboard/{name_target}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    args.dims_layers_ae = [data.shape[1]] + args.dims_layers_ae
    model_ae = AutoEncoder(args.dims_layers_ae, args.use_dropout).to(device)

    criterion_ae = nn.MSELoss()
    optimizer = torch.optim.Adam(model_ae.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 0.95)

    min_val_loss = np.Inf
    epochs_no_improve = 0
    fit_time_ae = 0
    writer = SummaryWriter(save_dir)
    model_path = f'{PurePosixPath(args.savefile).parent}/models_AE/{name_target}.pth'
    Path(PurePosixPath(model_path).parent).mkdir(parents=True, exist_ok=True)
    epoch_tqdm = tqdm(range(args.pretrain_epochs), desc="Epoch loss")
    for epoch in epoch_tqdm:
        loss_train, fit_t = train_step(model_ae, criterion_ae, optimizer, scheduler,
                                       data, labels, device, writer, epoch, args.batch_size)
        fit_time_ae += fit_t
        if loss_train < min_val_loss:
            torch.save(model_ae.state_dict(), model_path)
            epochs_no_improve = 0
            min_val_loss = loss_train
        else:
            epochs_no_improve += 1
        epoch_tqdm.set_description(
            f"Epoch loss: {loss_train:.5f} (minimal loss: {min_val_loss:.5f}, stop: {epochs_no_improve}|{args.earlyStopping})")
        if args.earlyStopping is not None and epoch > args.earlyStopping and epochs_no_improve == args.earlyStopping:
            print('\033[1;31mEarly stopping in AE model\033[0m')
            break

    print('===================================================')
    print(f'Transforming data to lower dimensional')
    if device.type == "cpu":
        model_ae.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    else:
        model_ae.load_state_dict(torch.load(model_path))
    model_ae.eval()

    low_data = np.empty((data.shape[0], args.dims_layers_ae[-1]))
    n_batch, rest = divmod(data.shape[0], args.batch_size)
    n_batch = n_batch + 1 if rest else n_batch
    score_time_ae = 0
    with torch.no_grad():
        test_tqdm = tqdm(range(n_batch), desc="Transform data", leave=False)
        for i in test_tqdm:
            start_time = time.time()
            batch = torch.from_numpy(data[i * args.batch_size:(i + 1) * args.batch_size, :]).float().to(device)
            # ===================forward=====================
            z, _ = model_ae(batch)
            low_data[i * args.batch_size:(i + 1) * args.batch_size, :] = z.detach().cpu().numpy()
            end_time = time.time()
            score_time_ae += end_time - start_time
    print('Data shape after transformation: {}'.format(low_data.shape))
    print('===================================================')

    if args.class_weight:
        args.class_weight = 'balanced'
    else:
        args.class_weight = None

    # Split data
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.1, random_state=args.seed)
    scoring = {'acc': make_scorer(accuracy_score), 'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
               'mcc': make_scorer(matthews_corrcoef), 'bal': make_scorer(balanced_accuracy_score),
               'recall': make_scorer(recall_score)}

    max_iters = 10000
    save_results(args.savefile, 'w', 'model', None, True, fit_time_ae=fit_time_ae, score_time_ae=score_time_ae)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        environ["PYTHONWARNINGS"] = "ignore"

        # Linear SVM
        print("\rLinear SVM         ", end='')
        parameters = {'C': [0.01, 0.1, 1, 10, 100]}
        # svc = svm.LinearSVC(class_weight=args.class_weight, random_state=seed)
        svc = svm.SVC(kernel='linear', class_weight=args.class_weight, random_state=args.seed, probability=True,
                      max_iter=max_iters)
        clf = GridSearchCV(svc, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc',
                           return_train_score=True)
        try:
            clf.fit(low_data, labels)
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)

        save_results(args.savefile, 'a', 'Linear SVM', clf, False, fit_time_ae=fit_time_ae, score_time_ae=score_time_ae)

        # RBF SVM
        print("\rRBF SVM             ", end='')
        parameters = {'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 1e-2, 1e-3, 1e-4]}
        svc = svm.SVC(gamma="scale", class_weight=args.class_weight, random_state=args.seed, probability=True,
                      max_iter=max_iters)
        clf = GridSearchCV(svc, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc',
                           return_train_score=True)
        try:
            clf.fit(low_data, labels)
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
        save_results(args.savefile, 'a', 'RBF SVM', clf, False, fit_time_ae=fit_time_ae, score_time_ae=score_time_ae)

        # LogisticRegression
        print("\rLogisticRegression  ", end='')
        lreg = LogisticRegression(random_state=args.seed, solver='lbfgs', multi_class='ovr',
                                  class_weight=args.class_weight,
                                  n_jobs=-1, max_iter=max_iters)
        parameters = {'C': [0.01, 0.1, 1, 10, 100]}
        clf = GridSearchCV(lreg, parameters, cv=sss, n_jobs=-1, scoring=scoring, refit='roc_auc',
                           return_train_score=True)
        try:
            clf.fit(low_data, labels)
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
        save_results(args.savefile, 'a', 'LogisticRegression', clf, False, fit_time_ae=fit_time_ae,
                     score_time_ae=score_time_ae)
        print()


if __name__ == '__main__':
    main()
