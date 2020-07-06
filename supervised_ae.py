import argparse
# from datetime import datetime
from pathlib import Path, PurePosixPath
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn.model_selection import StratifiedShuffleSplit
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm

from data import read_data
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


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


class Classifier(nn.Module):
    def __init__(self, dims_layers):
        super(Classifier, self).__init__()
        self.output = nn.Sequential()
        for i in range(1, len(dims_layers)):
            self.output.add_module('dense{:d}'.format(i), nn.Linear(dims_layers[i - 1], dims_layers[i]))
            self.output.add_module('reslu{:d}'.format(i), nn.ReLU(True))
        self.output.add_module('dense{:d}'.format(len(dims_layers)), nn.Linear(dims_layers[-1], 1))
        self.output.add_module('sigmoid', nn.Sigmoid())

    def forward(self, z):
        return self.output(z)


def train_step(model_AE, model_CL, criterion_AE, criterion_CL, optimizer, data_train, target, device,
               writer_tensorboard, n_epoch, batch_size, stage, scale_loss=1.):
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


def smooth(y, window_size=9, poly_order=5):
    # interpolate + smooth
    y = np.array(y)
    itp = interp1d(np.arange(y.size), y, kind='linear')
    return savgol_filter(itp(np.linspace(0, y.size - 1, 1000)), window_size, poly_order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_act', required=True, help='Path to data.')
    parser.add_argument('--data_in', default=None, help='Path to data.')
    parser.add_argument('--data_zinc', default=None, help='Path to data.')
    parser.add_argument('--scale_zinc', type=int, default=1)

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

    parser.add_argument('--dir_save', default='./outputs', help='Path to dictionary where will be save results.')
    parser.add_argument('--ae', default=None, help='Path to saved model.')
    parser.add_argument('--classifier', default=None, help='Path to saved model.')
    parser.add_argument('--test', action='store_true', help='Test model')
    parser.add_argument('--procedure', nargs='+', choices=['pre-training_ae', 'training_classifier', 'training_all'],
                        help='Procedure which you can use. Choice from: pre-training_ae, training_all, '
                             'training_classifier')
    parser.add_argument('--criterion_classifier', default='BCELoss', choices=['BCELoss', 'HingeLoss'],
                        help='Kind of loss function')
    parser.add_argument('--scale_loss', type=float, default=1., help='Weight for loss of classifier')
    args = parser.parse_args()

    # current_date = datetime.now()
    # current_date = current_date.strftime('%d%b_%H%M%S')
    dir_save = '{}/tensorboard/{}'.format(args.dir_save, PurePosixPath(args.data_act).stem.replace('_act', ''))
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
        labels = np.concatenate((labels, np.zeros(data.shape[0] - labels.size)))
    elif args.data_in is not None:
        # args.data_act = './on_bits/A2A_act_onbits'
        # args.data_in = './on_bits/A2A_in_onbits'

        data = read_data(args.data_act)
        labels = np.ones(data.shape[0])
        data = np.concatenate((data, read_data(args.data_in)), axis=0)
        labels = np.concatenate((labels, np.zeros(data.shape[0] - labels.size)))
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
    assert args.dims_layers_ae[-1] == args.dims_layers_classifier[0], 'Dimension of latent space must be equal with ' \
                                                                      'dimension of input classifier!'

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

    model_classifier = Classifier(args.dims_layers_classifier).to(device)
    if args.classifier is not None:
        if device.type == "cpu":
            model_classifier.load_state_dict(torch.load(args.classifier, map_location=lambda storage, loc: storage))
        else:
            model_classifier.load_state_dict(torch.load(args.classifier))
        model_classifier.eval()

    criterion_ae = nn.MSELoss()
    if args.criterion_classifier == 'HingeLoss':
        criterion_classifier = nn.HingeEmbeddingLoss()
        print('Use "Hinge" loss.')
    else:
        criterion_classifier = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(list(model_ae.parameters()) + list(model_classifier.parameters()), lr=args.lr,
                                 weight_decay=1e-5)

    writer = SummaryWriter(logdir=dir_save)
    if 'pre-training_ae' in args.procedure:
        epoch_tqdm = tqdm(range(args.pretrain_epochs), desc="Epoch pre-train loss")
        for epoch in epoch_tqdm:
            loss_train = train_step(model_ae, model_classifier, criterion_ae, criterion_classifier, optimizer,
                                    x_train, y_train, device, writer, epoch, args.batch_size, 'pre-training_ae')
            loss_test = test_step(model_ae, model_classifier, criterion_ae, criterion_classifier, x_test, y_test,
                                  device, writer, epoch, args.batch_size, 'pre-training_ae')
            epoch_tqdm.set_description("Epoch pre-train loss: {:.5f}, test loss: {:.5f}".format(loss_train, loss_test))

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
        epoch_tqdm.set_description("Epoch train loss: {:.5f}, test loss: {:.5f}".format(loss_train, loss_test))
        acc.append(acc_value)
        roc_auc.append(ra)
        m_corr.append(mc)
    writer.close()
    torch.save(model_ae.state_dict(), '{}/ae_model-{}.pth'.format(args.dir_save, PurePosixPath(args.data_act).stem.replace('_act', '')))
    torch.save(model_classifier.state_dict(), '{}/classifier_model-{}.pth'.format(args.dir_save, PurePosixPath(args.data_act).stem.replace('_act', '')))

    save_file = '{}/{}'.format(args.dir_save, PurePosixPath(args.data_act).stem.replace('_act', ''))

    plt.figure(figsize=(10, 8))
    x = np.arange(args.epochs)
    plt.plot(x, acc, linewidth=3, markersize=5, color='r', marker='o', label='acc')
    plt.plot(x, roc_auc, linewidth=3, markersize=5, color='g', marker='o', label='roc_auc')
    plt.plot(x, m_corr, linewidth=3, markersize=5, color='b', marker='o', label='mcc')
    plt.xlabel('Number of epoch', size=14)  # nazwa osi X
    plt.ylabel('Score', size=14)  # nazwa osi Y
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig('{}.png'.format(save_file), bbox_inches='tight', pad_inches=0)

    with open('{}.txt'.format(save_file), 'a') as f:
        idx = int(np.argmax(acc))
        # f.write('{:.4f} {:.4f}\n'.format(acc[idx], np.max(smooth(acc))))
        f.write('{:.4f} | ACC\n'.format(acc[idx]))
        f.write('{:.4f} | ROC_AUC\n'.format(roc_auc[idx]))
        f.write('{:.4f} | MCC\n'.format(m_corr[idx]))


if __name__ == '__main__':
    main()
