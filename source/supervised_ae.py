import argparse
# from datetime import datetime
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedShuffleSplit
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm

from scipy.sparse import load_npz

from models import AutoEncoder, Classifier


def train_step(model_AE, model_CL, criterion_AE, criterion_CL, optimizer, data_train, target, device,
               writer_tensorboard, n_epoch, batch_size, stage, scale_loss=1.):
    if model_AE is not None:
        model_AE.train()
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

    train_tqdm = tqdm(range(n_batch), desc="Train loss", leave=False)
    for i in train_tqdm:
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

        loss = loss_ae + scale_loss * loss_cl
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
    return loss.item()


def test_step(model_AE, model_CL, criterion_AE, criterion_CL, data_test, target, device,
              writer_tensorboard, n_epoch, batch_size, stage, scale_loss=1.):
    if model_AE is not None:
        model_AE.eval()
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

    with torch.no_grad():
        test_tqdm = tqdm(range(n_batch), desc="Test loss", leave=False)
        for i in test_tqdm:
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
                output_class.append(output_cl.round().detach().cpu().numpy())
            else:
                loss_ae = criterion_AE(output_ae, batch)
                output_cl = model_CL(z)
                loss_cl = criterion_CL(output_cl, labels)

                elements += labels.size(0)
                accuracy += (output_cl.round() == labels).sum().item()

                label_class.append(target[batch_idx])
                output_class.append(output_cl.round().detach().cpu().numpy())

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
        writer_tensorboard.add_scalar('pre_train/tst_ae', test_loss_ae, n_epoch)
        return test_loss
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
    output_class = np.concatenate(output_class, axis=0)
    label_class = np.concatenate(label_class, axis=0)
    ra = roc_auc_score(label_class, output_class)
    mc = matthews_corrcoef(label_class, output_class)
    return test_loss, accuracy, ra, mc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/data', help='Name/path of target')
    parser.add_argument('--name', default=['5HT2A'], nargs='+', help='Name of target')

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

    parser.add_argument('--save_dir', default='./outputs', help='Path to dictionary where will be save results.')
    parser.add_argument('--ae', default=None, help='Path to saved model.')
    parser.add_argument('--classifier', default=None, help='Path to saved model.')
    parser.add_argument('--test', action='store_true', help='Test model')
    parser.add_argument('--procedure', nargs='+', choices=['pre-training_ae', 'training_classifier', 'training_all'],
                        help='Procedure which you can use. Choice from: pre-training_ae, training_all, '
                             'training_classifier')
    parser.add_argument('--criterion_classifier', default='BCELoss', choices=['BCELoss', 'HingeLoss'],
                        help='Kind of loss function')
    parser.add_argument('--scale_loss', type=float, default=1., help='Weight for loss of classifier')
    parser.add_argument('--save_model', action='store_true', help='Save all models')
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

    if args.dims_layers_classifier[0] == -1:
        args.dims_layers_classifier[0] = data.shape[1]

    model_classifier = Classifier(args.dims_layers_classifier, args.use_dropout).to(device)
    if args.classifier is not None:
        print(f"\033[1;5;33mLoad model CLR form '{args.classifier}'\033[0m")
        if device.type == "cpu":
            model_classifier.load_state_dict(torch.load(args.classifier, map_location=lambda storage, loc: storage))
        else:
            model_classifier.load_state_dict(torch.load(args.classifier))
        model_classifier.eval()

    if args.criterion_classifier == 'HingeLoss':
        criterion_classifier = nn.HingeEmbeddingLoss()
        print('Use "Hinge" loss.')
    else:
        criterion_classifier = nn.BCEWithLogitsLoss()

    model_ae = None
    criterion_ae = None
    if 'training_classifier' != args.procedure[0]:
        args.dims_layers_ae = [data.shape[1]] + args.dims_layers_ae
        assert args.dims_layers_ae[-1] == args.dims_layers_classifier[0], \
            'Dimension of latent space must be equal with dimension of input classifier!'

        model_ae = AutoEncoder(args.dims_layers_ae, args.use_dropout).to(device)
        if args.ae is not None:
            print(f"\033[1;5;33mLoad model AE form '{args.ae}'\033[0m")
            if device.type == "cpu":
                model_ae.load_state_dict(torch.load(args.ae, map_location=lambda storage, loc: storage))
            else:
                model_ae.load_state_dict(torch.load(args.ae))
            model_ae.eval()

        criterion_ae = nn.MSELoss()

        optimizer = torch.optim.Adam(list(model_ae.parameters()) + list(model_classifier.parameters()), lr=args.lr,
                                     weight_decay=1e-5)
    else:
        optimizer = torch.optim.Adam(model_classifier.parameters(), lr=args.lr, weight_decay=1e-5)

    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=0)
    for train_index, test_index in sss.split(data, labels):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        del train_index, test_index
        break

    writer = SummaryWriter(logdir=save_dir)
    if 'pre-training_ae' in args.procedure:
        epoch_tqdm = tqdm(range(args.pretrain_epochs), desc="Epoch pre-train loss")
        for epoch in epoch_tqdm:
            loss_train = train_step(model_ae, model_classifier, criterion_ae, criterion_classifier, optimizer,
                                    x_train, y_train, device, writer, epoch, args.batch_size, 'pre-training_ae')
            loss_test = test_step(model_ae, model_classifier, criterion_ae, criterion_classifier, x_test, y_test,
                                  device, writer, epoch, args.batch_size, 'pre-training_ae')
            epoch_tqdm.set_description(f"Epoch pre-train loss: {loss_train:.5f}, test loss: {loss_test:.5f}")

    acc = []
    roc_auc = []
    m_corr = []
    stage = 'training_classifier' if 'training_classifier' in args.procedure else 'training_all'
    epoch_tqdm = tqdm(range(args.epochs), desc="Epoch train loss")
    for epoch in epoch_tqdm:
        loss_train = train_step(model_ae, model_classifier, criterion_ae, criterion_classifier, optimizer,
                                x_train, y_train, device, writer, epoch, args.batch_size, stage, args.scale_loss)
        loss_test, acc_value, ra, mc = test_step(model_ae, model_classifier, criterion_ae, criterion_classifier,
                                                 x_test, y_test, device, writer, epoch, args.batch_size,
                                                 stage, args.scale_loss)
        epoch_tqdm.set_description(f"Epoch train loss: {loss_train:.5f}, test loss: {loss_test:.5f}")
        acc.append(acc_value)
        roc_auc.append(ra)
        m_corr.append(mc)
    writer.close()
    if args.save_model:
        if model_ae is not None:
            torch.save(model_ae.state_dict(), f'{args.save_dir}/ae_model-{args.name[id_file_target]}.pth')
        torch.save(model_classifier.state_dict(), f'{args.save_dir}/classifier_model-{args.name[id_file_target]}.pth')

    save_file = f'{args.save_dir}/{args.name[id_file_target]}'

    plt.figure(figsize=(10, 8))
    x = np.arange(args.epochs)
    plt.plot(x, acc, linewidth=3, markersize=5, color='r', marker='o', label='acc')
    plt.plot(x, roc_auc, linewidth=3, markersize=5, color='g', marker='o', label='roc_auc')
    plt.plot(x, m_corr, linewidth=3, markersize=5, color='b', marker='o', label='mcc')
    plt.xlabel('Number of epoch', size=14)  # nazwa osi X
    plt.ylabel('Score', size=14)  # nazwa osi Y
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig(f'{save_file}.png', bbox_inches='tight', pad_inches=0)

    with open(f'{save_file}.txt', 'a') as f:
        idx = int(np.argmax(roc_auc))
        f.write('{:.4f} | ACC\n'.format(acc[idx]))
        f.write('{:.4f} | ROC_AUC\n'.format(roc_auc[idx]))
        f.write('{:.4f} | MCC\n'.format(m_corr[idx]))


if __name__ == '__main__':
    main()
