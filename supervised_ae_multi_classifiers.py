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
from functools import reduce

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


def train_step(model_AE, models_CL, criterion_AE, criterion_CL, optimizer_AE, optimizers_CL, data_train, target, device,
               writer_tensorboard, n_epoch, batch_size, stage, scale: float = 1):
    model_AE.train()
    batch_iters = []
    max_num_batch = 0
    for i in range(len(models_CL)):
        models_CL[i].train()
        gen = GetBatch(target[i], batch_size)
        batch_iters.append(gen())
        if max_num_batch < len(gen):
            max_num_batch = len(gen)

    accuracy = [0 for _ in range(len(models_CL))]
    train_tqdm = tqdm(range(max_num_batch), desc="Train loss", leave=False)
    for b in train_tqdm:
        loss_ae_all = torch.tensor([0]).float().to(device)
        loss_cl = []
        loss_cl_all = 0
        for i in range(len(models_CL)):
            try:
                batch_idx = next(batch_iters[i])
            except StopIteration:
                batch_iters[i] = GetBatch(target[i], batch_size)()
                batch_idx = next(batch_iters[i])

            batch = torch.from_numpy(data_train[i][batch_idx]).float().to(device)
            labels = torch.from_numpy(target[i][batch_idx]).float().view(-1, 1).to(device)
            # ===================forward=====================
            z, output_ae = model_AE(batch)

            if stage == 'pre-training_ae':
                loss_ae = criterion_AE(output_ae, batch)
            elif stage == 'training_classifier':
                loss_ae = torch.tensor([0]).float().to(device)
                output_cl = models_CL[i](z)
                loss_cl.append(criterion_CL(output_cl, labels))
                accuracy[i] = (output_cl.round() == labels).sum().item() / labels.size(0)
                loss_cl_all += loss_cl[i].item()
                # =================== backward classifiers ====================
                optimizers_CL[i].zero_grad()
                loss_cl[i].backward()
                optimizers_CL[i].step()
            else:
                loss_ae = criterion_AE(output_ae, batch)
                # output_cl = models_CL[i](z.detach())
                output_cl = models_CL[i](z)
                loss_cl.append(criterion_CL(output_cl, labels))
                accuracy[i] = (output_cl.round() == labels).sum().item() / labels.size(0)
                loss_cl_all += loss_cl[i].item()
                # ===================backward====================
                optimizer_AE.zero_grad()
                (loss_ae + scale * loss_cl[i]).backward()
                optimizer_AE.step()
            loss_ae_all = loss_ae_all + loss_ae
        # ===================backward====================
        if stage == 'pre-training_ae':
            optimizer_AE.zero_grad()
            loss_ae_all.backward()
            optimizer_AE.step()
        # ===================log========================
        train_tqdm.set_description(
            "Train loss AE: {:.5f}, loss classifier: {:.5f}".format(loss_ae_all.item() / len(models_CL),
                                                                    loss_cl_all / len(models_CL)))
        if stage == 'pre-training_ae':
            writer_tensorboard.add_scalar('pre_train/trn_ae', loss_ae_all / len(models_CL),
                                          n_epoch * max_num_batch + b)
        elif stage == 'training_classifier':
            for i in range(len(models_CL)):
                writer_tensorboard.add_scalar('train_trn_classifier/{:d}'.format(i), loss_cl[i],
                                              n_epoch * max_num_batch + b)
                writer_tensorboard.add_scalar('train_trn_acc/{:d}'.format(i), accuracy[i], n_epoch * max_num_batch + b)
        else:
            for i in range(len(models_CL)):
                writer_tensorboard.add_scalar('train_trn_classifier/{:d}'.format(i), loss_cl[i],
                                              n_epoch * max_num_batch + b)
                writer_tensorboard.add_scalar('train_trn_acc/{:d}'.format(i), accuracy[i], n_epoch * max_num_batch + b)
            writer_tensorboard.add_scalar('train/trn_ae', loss_ae, n_epoch * max_num_batch + b)


def test_step(model_AE, models_CL, criterion_AE, criterion_CL, data_test, targets, device, writer_tensorboard, n_epoch,
              batch_size, stage):
    model_AE.eval()
    batch_iters = []
    num_batch = np.zeros(len(models_CL), dtype=int)
    for i in range(len(models_CL)):
        models_CL[i].train()
        gen = GetBatch(targets[i], batch_size)
        batch_iters.append(gen())
        num_batch[i] = len(gen)
    max_num_batch = num_batch.max()

    test_loss_ae = 0
    test_loss_cl = [0 for _ in range(len(models_CL))]

    accuracy = [0 for _ in range(len(models_CL))]
    elements = [0 for _ in range(len(models_CL))]

    label_class = [[] for _ in range(len(models_CL))]
    output_class = [[] for _ in range(len(models_CL))]

    with torch.no_grad():
        for _ in tqdm(range(max_num_batch), leave=False):
            for i in range(len(models_CL)):
                try:
                    batch_idx = next(batch_iters[i])
                except StopIteration:
                    continue

                batch = torch.from_numpy(data_test[i][batch_idx]).float().to(device)
                labels = torch.from_numpy(targets[i][batch_idx]).float().view(-1, 1).to(device)
                # ===================forward=====================
                z, output_ae = model_AE(batch)

                if stage == 'pre-training_ae':
                    loss_ae = criterion_AE(output_ae, batch)
                    loss_cl = torch.tensor([0]).float().to(device)
                elif stage == 'training_classifier':
                    loss_ae = torch.tensor([0]).float().to(device)
                    output_cl = models_CL[i](z)
                    loss_cl = criterion_CL(output_cl, labels)
    
                    elements[i] += labels.size(0)
                    accuracy[i] += (output_cl.round() == labels).sum().item()
    
                    label_class[i].append(targets[i][batch_idx])
                    output_class[i].append(output_cl.round().detach().cpu().numpy())
                else:
                    loss_ae = criterion_AE(output_ae, batch)
                    output_cl = models_CL[i](z)
                    loss_cl = criterion_CL(output_cl, labels)
    
                    elements[i] += labels.size(0)
                    accuracy[i] += (output_cl.round() == labels).sum().item()
    
                    label_class[i].append(targets[i][batch_idx])
                    output_class[i].append(output_cl.round().detach().cpu().numpy())

                test_loss_ae += loss_ae.item()
                test_loss_cl[i] += loss_cl.item()
    # ===================log========================

    test_loss_ae /= num_batch.sum()
    test_loss_cl = [test_loss_cl[i] / num_batch[i] for i in range(len(models_CL))]
    if stage == 'pre-training_ae':
        writer_tensorboard.add_scalar('pre_train/tst_ae', test_loss_ae, n_epoch)
    elif stage == 'training_classifier':
        for i in range(len(models_CL)):
            accuracy[i] /= elements[i]
            writer_tensorboard.add_scalar('train_tst_classifier/{:d}'.format(i), test_loss_cl[i], n_epoch)
            writer_tensorboard.add_scalar('train_tst_acc/{:d}'.format(i), accuracy[i], n_epoch)
    else:
        for i in range(len(models_CL)):
            accuracy[i] /= elements[i]
            writer_tensorboard.add_scalar('train/tst_ae', test_loss_ae, n_epoch)
            writer_tensorboard.add_scalar('train_tst_classifier/{:d}'.format(i), test_loss_cl[i], n_epoch)
            writer_tensorboard.add_scalar('train_tst_acc/{:d}'.format(i), accuracy[i], n_epoch)
    if stage != 'pre-training_ae':
        ra = [0 for _ in range(len(models_CL))]
        mc = [0 for _ in range(len(models_CL))]
        for i in range(len(models_CL)):
            output_cl = np.concatenate(output_class[i], axis=0)
            label_cl = np.concatenate(label_class[i], axis=0)
            ra[i] = roc_auc_score(label_cl, output_cl)
            mc[i] = matthews_corrcoef(label_cl, output_cl)
        return accuracy, ra, mc


def smooth(y, window_size=9, poly_order=5):
    # interpolate + smooth
    y = np.array(y)
    itp = interp1d(np.arange(y.size), y, kind='linear')
    return savgol_filter(itp(np.linspace(0, y.size - 1, 1000)), window_size, poly_order)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_act', nargs='+', required=True, help='Path to data.')
    parser.add_argument('--data_in', nargs='+', default=None, help='Path to data.')
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
    parser.add_argument('--classifiers', nargs='+', default=None, help='Path to saved model.')
    parser.add_argument('--test', action='store_true', help='Test model')
    parser.add_argument('--procedure', nargs='+', choices=['pre-training_ae', 'training_classifier', 'training_all'],
                        help='Procedure which you can use. Choice from: pre-training_ae, training_all, '
                             'training_classifier')
    parser.add_argument('--criterion_classifier', default='BCELoss', choices=['BCELoss', 'HingeLoss'],
                        help='Kind of loss function')
    parser.add_argument('--scale', type=float, default=1, help='Scale of cost of classifier')
    args = parser.parse_args()

    # current_date = datetime.now()
    # current_date = current_date.strftime('%d%b_%H%M%S')
    dir_save = '{}/tensorboard'.format(args.dir_save)
    save_file = '{}/results'.format(args.dir_save)
    Path(dir_save).mkdir(parents=True, exist_ok=True)
    Path(save_file).mkdir(parents=True, exist_ok=True)

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
    assert args.dims_layers_ae[-1] == args.dims_layers_classifier[0], 'Dimension of latent space must be equal with ' \
                                                                      'dimension of input classifier!'

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

    models_classifier = []
    for i in range(num_datasets):
        models_classifier.append(Classifier(args.dims_layers_classifier).to(device))
        if args.classifiers is not None:
            if device.type == "cpu":
                models_classifier[-1].load_state_dict(torch.load(args.classifiers[i],
                                                                 map_location=lambda storage, loc: storage))
            else:
                models_classifier[-1].load_state_dict(torch.load(args.classifiers[i]))
            models_classifier[-1].eval()

    criterion_ae = nn.MSELoss()
    if args.criterion_classifier == 'HingeLoss':
        criterion_classifier = nn.HingeEmbeddingLoss()
        print('Use "Hinge" loss.')
    else:
        criterion_classifier = nn.BCEWithLogitsLoss()

    optimizer_AE = torch.optim.Adam(list(model_ae.parameters()), lr=args.lr, weight_decay=1e-5)
    optimizers_CL = []
    for i in range(num_datasets):
        optimizers_CL.append(torch.optim.Adam(list(models_classifier[i].parameters()), lr=args.lr, weight_decay=1e-5))
        
    params_classifiers = []
    for i in range(num_datasets):
        params_classifiers += list(models_classifier[i].parameters())
    optimizer_all = torch.optim.Adam(list(model_ae.parameters()) + params_classifiers, lr=args.lr, weight_decay=1e-5)

    writer = SummaryWriter(logdir=dir_save)
    if 'pre-training_ae' in args.procedure:
        for epoch in tqdm(range(args.pretrain_epochs)):
            train_step(model_ae, models_classifier, criterion_ae, criterion_classifier, optimizer_AE, optimizers_CL,
                       x_train, y_train, device, writer, epoch, args.batch_size, 'pre-training_ae', args.scale)
            test_step(model_ae, models_classifier, criterion_ae, criterion_classifier, x_test, y_test, device, writer,
                      epoch, args.batch_size, 'pre-training_ae')

    acc = [[] for _ in range(num_datasets)]
    roc_auc = [[] for _ in range(num_datasets)]
    m_corr = [[] for _ in range(num_datasets)]
    stage = 'training_classifier' if 'training_classifier' in args.procedure else 'training_all'
    for epoch in tqdm(range(args.epochs)):
        train_step(model_ae, models_classifier, criterion_ae, criterion_classifier,
                   optimizer_AE if stage != 'training_all' else optimizer_all,
                   optimizers_CL, x_train, y_train, device, writer, epoch, args.batch_size, stage, args.scale)
        acc_value, ra, mc = test_step(model_ae, models_classifier, criterion_ae, criterion_classifier, x_test, y_test,
                                      device, writer, epoch, args.batch_size, stage)
        for i in range(num_datasets):
            acc[i].append(acc_value[i])
            roc_auc[i].append(ra[i])
            m_corr[i].append(mc[i])
    writer.close()
    torch.save(model_ae.state_dict(), '{}/ae_model.pth'.format(args.dir_save))
    for i in range(num_datasets):
        torch.save(models_classifier[i].state_dict(), '{}/classifier_model_{}.pth'.format(args.dir_save, i))

    x = np.arange(args.epochs)
    fig = plt.figure(figsize=(10, 8))
    for i in range(num_datasets):
        plt.plot(x, acc[i], linewidth=3, markersize=5, color='r', marker='o', label='acc[{:d}]'.format(i))
    plt.xlabel('Number of epoch', size=14)  # nazwa osi X
    plt.ylabel('acc', size=14)  # nazwa osi Y
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig('{}/acc.png'.format(save_file), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    fig = plt.figure(figsize=(10, 8))
    for i in range(num_datasets):
        plt.plot(x, roc_auc[i], linewidth=3, markersize=5, color='g', marker='o', label='roc_auc')
    plt.xlabel('Number of epoch', size=14)  # nazwa osi X
    plt.ylabel('roc_auc', size=14)  # nazwa osi Y
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig('{}/roc_auc.png'.format(save_file), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    fig = plt.figure(figsize=(10, 8))
    for i in range(num_datasets):
        plt.plot(x, m_corr[i], linewidth=3, markersize=5, color='b', marker='o', label='mcc')
    plt.xlabel('Number of epoch', size=14)  # nazwa osi X
    plt.ylabel('m_corr', size=14)  # nazwa osi Y
    plt.legend()
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.savefig('{}/m_corr.png'.format(save_file), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    with open('{}/scores.txt'.format(save_file), 'w') as f:
        for i in range(num_datasets):
            idx = int(np.argmax(acc[i]))
            f.write('{}:\n'.format(PurePosixPath(args.data_act[i]).stem.replace('_act', '')))
            f.write('\t{:.4f} | ACC\n'.format(acc[i][idx]))
            f.write('\t{:.4f} | ROC_AUC\n'.format(roc_auc[i][idx]))
            f.write('\t{:.4f} | MCC\n'.format(m_corr[i][idx]))


if __name__ == '__main__':
    main()
