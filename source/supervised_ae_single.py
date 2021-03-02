import argparse
import time
import warnings
from os import environ
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, balanced_accuracy_score, recall_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import AutoEncoder, Classifier


def train_step(model_AE, model_CL, criterion_AE, criterion_CL, optimizer, scheduler, data_train, target, device,
               writer_tensorboard, n_epoch, batch_size, stage, scale_loss=1.):
    if model_AE is not None:
        model_AE.train()
    if model_CL is not None:
        model_CL.train()

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
        labels = torch.from_numpy(target[batch_idx]).float().view(-1, 1).to(device)
        # ===================forward=====================
        if model_AE is None:
            z = batch
        else:
            z, output_ae = model_AE(batch)

        if stage == 'pre-training_ae':
            loss_ae = criterion_AE(output_ae, batch)
            loss_cl = torch.tensor([0]).float().to(device)
        elif stage == 'training_classifier':
            loss_ae = torch.tensor([0]).float().to(device)
            output_cl = model_CL(z)
            loss_cl = criterion_CL(output_cl, labels)
            accuracy = (output_cl.round() == labels).sum().item() / labels.size(0)
        else:
            loss_ae = criterion_AE(output_ae, batch)
            output_cl = model_CL(z)
            loss_cl = criterion_CL(output_cl, labels)
            accuracy = (output_cl.round() == labels).sum().item() / labels.size(0)
        # assert torch.isfinite(loss_ae), '\033[1;31mLoss AE has Nan or Inf\033[0m'
        # assert torch.isfinite(loss_cl), '\033[1;31mLoss for classifier has Nan or Inf\033[0m'
        loss = loss_ae + scale_loss * loss_cl
        if not torch.isfinite(loss):
            print('\033[1;31mLoss has Nan or Inf\033[0m')
            break
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        end_time = time.time()
        time_work += end_time - start_time
        # ===================log========================
        train_tqdm.set_description(
            "Train loss AE: {:.5f}, loss classifier: {:.5f}".format(loss_ae.item(), loss_cl.item()))
        if stage == 'pre-training_ae':
            writer_tensorboard.add_scalar('pre_train/trn_ae', loss_ae, n_epoch * n_batch + i)
        elif stage == 'training_classifier':
            writer_tensorboard.add_scalar('train/trn_classifier', loss_cl, n_epoch * n_batch + i)
            writer_tensorboard.add_scalar('train/trn_acc', accuracy, n_epoch * n_batch + i)
        else:
            writer_tensorboard.add_scalar('train/trn_total', loss, n_epoch * n_batch + i)
            writer_tensorboard.add_scalar('train/trn_ae', loss_ae, n_epoch * n_batch + i)
            writer_tensorboard.add_scalar('train/trn_classifier', loss_cl, n_epoch * n_batch + i)
            writer_tensorboard.add_scalar('train/trn_acc', accuracy, n_epoch * n_batch + i)
    return loss.item(), time_work


def test_step(model_AE, model_CL, criterion_AE, criterion_CL, data_test, target, device,
              writer_tensorboard, n_epoch, batch_size, stage, scale_loss=1.):
    if model_AE is not None:
        model_AE.eval()
    if model_CL is not None:
        model_CL.eval()
    test_loss = 0
    test_loss_ae = 0
    test_loss_cl = 0

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

    accuracy = 0.
    elements = 0

    label_class = []
    output_class = []

    time_work = 0
    with torch.no_grad():
        test_tqdm = tqdm(range(n_batch), desc="Test loss", leave=False)
        for i in test_tqdm:
            start_time = time.time()
            batch_idx = active_id[i * batch_size:(i + 1) * batch_size]
            batch_idx = np.concatenate((batch_idx, inactive_id[i * batch_size:(i + 1) * batch_size]), axis=0)

            idx = np.random.permutation(np.arange(batch_idx.size))
            batch_idx = batch_idx[idx]

            batch = torch.from_numpy(data_test[batch_idx]).float().to(device)
            labels = torch.from_numpy(target[batch_idx]).float().view(-1, 1).to(device)
            # ===================forward=====================
            if model_AE is None:
                z = batch
            else:
                z, output_ae = model_AE(batch)

            if stage == 'pre-training_ae':
                loss_ae = criterion_AE(output_ae, batch)
                loss_cl = torch.tensor([0]).float().to(device)
            elif stage == 'training_classifier':
                loss_ae = torch.tensor([0]).float().to(device)
                output_cl = model_CL(z)
                loss_cl = criterion_CL(output_cl, labels)

                elements += labels.size(0)
                accuracy += (output_cl.round() == labels).sum().item()

                label_class.append(target[batch_idx])
                output_class.append(output_cl.detach().cpu().numpy())
            else:
                loss_ae = criterion_AE(output_ae, batch)
                output_cl = model_CL(z)
                loss_cl = criterion_CL(output_cl, labels)

                elements += labels.size(0)
                accuracy += (output_cl.round() == labels).sum().item()

                label_class.append(target[batch_idx])
                output_class.append(output_cl.detach().cpu().numpy())
            end_time = time.time()
            time_work += end_time - start_time

            loss = loss_ae + scale_loss * loss_cl

            test_loss_ae += loss_ae.item()
            test_loss_cl += loss_cl.item()
            test_loss += loss.item()  # sum up batch loss
    # ===================log========================

    test_loss /= n_batch
    test_loss_ae /= n_batch
    test_loss_cl /= n_batch
    test_tqdm.set_description("Test loss: {:.5f}".format(test_loss))
    if stage == 'pre-training_ae':
        writer_tensorboard.add_scalar(f'pre_train/tst_ae', test_loss_ae, n_epoch)
        return test_loss, time_work
    elif stage == 'training_classifier':
        accuracy /= elements
        writer_tensorboard.add_scalar('train/tst_classifier', test_loss_cl, n_epoch)
        writer_tensorboard.add_scalar('train/tst_acc', accuracy, n_epoch)
    else:
        accuracy /= elements
        writer_tensorboard.add_scalar('train/tst_total', test_loss, n_epoch)
        writer_tensorboard.add_scalar('train/tst_ae', test_loss_ae, n_epoch)
        writer_tensorboard.add_scalar('train/tst_classifier', test_loss_cl, n_epoch)
        writer_tensorboard.add_scalar('train/tst_acc', accuracy, n_epoch)
    output_class = np.concatenate(output_class, axis=0).flatten()
    label_class = np.concatenate(label_class, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        environ["PYTHONWARNINGS"] = "ignore"
        scores = {'roc_auc': roc_auc_score(label_class, output_class)}
        output_class = np.around(output_class)
        scores['acc'] = accuracy_score(label_class, output_class)
        scores['mcc'] = matthews_corrcoef(label_class, output_class)
        scores['bal'] = balanced_accuracy_score(label_class, output_class)
        scores['recall'] = recall_score(label_class, output_class)
    return test_loss, scores, time_work


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', required=True, help='Name/path of file')
    parser.add_argument('--save_dir', default='./outputs', help='Path to dictionary where will be save results.')

    parser.add_argument('--pretrain_epochs', type=int, default=100, help="Number of epochs to pretrain model AE")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train AE and classifier")
    parser.add_argument('--dims_layers_ae', type=int, nargs='+', default=[500, 100, 10],
                        help="Dimensional of layers in AE")
    parser.add_argument('--dims_layers_classifier', type=int, nargs='+', default=[10, 5],
                        help="Dimensional of layers in classifier")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--use_dropout', action='store_true', help="Use dropout")

    parser.add_argument('--no-cuda', action='store_true', help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1)')

    parser.add_argument('--procedure', nargs='+', choices=['pre-training_ae', 'training_classifier', 'training_all'],
                        help='Procedure which you can use. Choice from: pre-training_ae, training_all, '
                             'training_classifier')
    parser.add_argument('--criterion_classifier', default='BCELoss', choices=['BCELoss', 'HingeLoss'],
                        help='Kind of loss function')
    parser.add_argument('--scale_loss', type=float, default=1., help='Weight for loss of classifier')
    parser.add_argument('--earlyStopping', type=int, default=None, help='Number of epochs to early stopping')
    parser.add_argument('--use_scheduler', action='store_true')
    args = parser.parse_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    loaded = np.load(args.filename)
    x_train = loaded[f'data_train']
    x_test = loaded[f'data_test']
    y_train = loaded[f'lab_train']
    y_test = loaded[f'lab_test']
    del loaded

    name_target = PurePosixPath(args.filename).parent.stem
    n_split = PurePosixPath(args.filename).stem
    save_dir = f'{args.save_dir}/tensorboard/{name_target}_{n_split}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if args.dims_layers_classifier[0] == -1:
        args.dims_layers_classifier[0] = x_test.shape[1]

    model_classifier = Classifier(args.dims_layers_classifier, args.use_dropout).to(device)
    if args.criterion_classifier == 'HingeLoss':
        criterion_classifier = nn.HingeEmbeddingLoss()
        print('Use "Hinge" loss.')
    else:
        criterion_classifier = nn.BCEWithLogitsLoss()

    model_ae = None
    criterion_ae = None
    if 'training_classifier' != args.procedure[0]:
        args.dims_layers_ae = [x_train.shape[1]] + args.dims_layers_ae
        assert args.dims_layers_ae[-1] == args.dims_layers_classifier[0], \
            'Dimension of latent space must be equal with dimension of input classifier!'

        model_ae = AutoEncoder(args.dims_layers_ae, args.use_dropout).to(device)
        criterion_ae = nn.MSELoss()
        optimizer = torch.optim.Adam(list(model_ae.parameters()) + list(model_classifier.parameters()), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model_classifier.parameters(), lr=args.lr)

    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: 0.95)

    writer = SummaryWriter(save_dir)

    total_scores = {'roc_auc': 0, 'acc': 0, 'mcc': 0, 'bal': 0, 'recall': 0,
                    'max_roc_auc': 0, 'max_acc': 0, 'max_mcc': 0, 'max_bal': 0, 'max_recall': 0,
                    'pre-fit_time': 0, 'pre-score_time': 0, 'fit_time': 0, 'score_time': 0
                    }

    dir_model_ae = f'{args.save_dir}/models_AE'
    Path(dir_model_ae).mkdir(parents=True, exist_ok=True)
    # dir_model_classifier = f'{args.save_dir}/models_classifier'
    # Path(dir_model_classifier).mkdir(parents=True, exist_ok=True)

    path_ae = f'{dir_model_ae}/{name_target}_{n_split}.pth'
    # path_classifier = f'{dir_model_classifier}/{name_target}_{n_split}.pth'

    if 'pre-training_ae' in args.procedure:
        min_val_loss = np.Inf
        epochs_no_improve = 0

        epoch_tqdm = tqdm(range(args.pretrain_epochs), desc="Epoch pre-train loss")
        for epoch in epoch_tqdm:
            loss_train, time_trn = train_step(model_ae, None, criterion_ae, None, optimizer, scheduler, x_train,
                                              y_train, device, writer, epoch, args.batch_size, 'pre-training_ae')
            loss_test, _ = test_step(model_ae, None, criterion_ae, None, x_test, y_test, device, writer, epoch,
                                     args.batch_size, 'pre-training_ae')

            if not np.isfinite(loss_train):
                break

            total_scores['pre-fit_time'] += time_trn

            if loss_test < min_val_loss:
                torch.save(model_ae.state_dict(), path_ae)
                epochs_no_improve = 0
                min_val_loss = loss_test
            else:
                epochs_no_improve += 1
            epoch_tqdm.set_description(
                f"Epoch pre-train loss: {loss_train:.5f}, test loss: {loss_test:.5f} (minimal val-loss: {min_val_loss:.5f}, stop: {epochs_no_improve}|{args.earlyStopping})")
            if args.earlyStopping is not None and epoch >= args.earlyStopping and epochs_no_improve == args.earlyStopping:
                print('\033[1;31mEarly stopping in pre-training model\033[0m')
                break
        print(f"\033[1;5;33mLoad model AE form '{path_ae}'\033[0m")
        if device.type == "cpu":
            model_ae.load_state_dict(torch.load(path_ae, map_location=lambda storage, loc: storage))
        else:
            model_ae.load_state_dict(torch.load(path_ae))
        model_ae = model_ae.to(device)
        model_ae.eval()

    min_val_loss = np.Inf
    epochs_no_improve = 0

    epoch = None
    stage = 'training_classifier' if 'training_classifier' in args.procedure else 'training_all'
    epoch_tqdm = tqdm(range(args.epochs), desc="Epoch train loss")
    for epoch in epoch_tqdm:
        loss_train, time_trn = train_step(model_ae, model_classifier, criterion_ae, criterion_classifier, optimizer,
                                          scheduler, x_train, y_train, device, writer, epoch, args.batch_size,
                                          stage, args.scale_loss)
        loss_test, scores_val, time_tst = test_step(model_ae, model_classifier, criterion_ae, criterion_classifier,
                                                    x_test, y_test, device, writer, epoch, args.batch_size, stage,
                                                    args.scale_loss)

        if not np.isfinite(loss_train):
            break

        total_scores['fit_time'] += time_trn
        total_scores['score_time'] += time_tst
        if total_scores['max_roc_auc'] < scores_val['roc_auc']:
            for key, val in scores_val.items():
                total_scores[f'max_{key}'] = val

        if loss_test < min_val_loss:
            # torch.save(model_ae.state_dict(), path_ae)
            # torch.save(model_classifier.state_dict(), path_classifier)
            epochs_no_improve = 0
            min_val_loss = loss_test
            for key, val in scores_val.items():
                total_scores[key] = val
        else:
            epochs_no_improve += 1
        epoch_tqdm.set_description(
            f"Epoch train loss: {loss_train:.5f}, test loss: {loss_test:.5f} (minimal val-loss: {min_val_loss:.5f}, stop: {epochs_no_improve}|{args.earlyStopping})")
        if args.earlyStopping is not None and epoch >= args.earlyStopping and epochs_no_improve == args.earlyStopping:
            print('\033[1;31mEarly stopping!\033[0m')
            break
    total_scores['score_time'] /= epoch + 1
    writer.close()

    save_file = f'{args.save_dir}/{name_target}.txt'
    head = 'idx;params'
    temp = f'{n_split};pretrain_epochs:{args.pretrain_epochs},dims_layers_ae:{args.dims_layers_ae},' \
           f'dims_layers_classifier:{args.dims_layers_classifier},batch_size:{args.batch_size},lr:{args.lr}' \
           f'use_dropout:{args.use_dropout},procedure:{args.procedure},scale_loss:{args.scale_loss},' \
           f'earlyStopping:{args.earlyStopping}'
    for key, val in total_scores.items():
        head = head + f';{key}'
        temp = temp + f';{val}'

    not_exists = not Path(save_file).exists()
    with open(save_file, 'a') as f:
        if not_exists:
            f.write(f'{head}\n')
        f.write(f'{temp}\n')


if __name__ == '__main__':
    main()
