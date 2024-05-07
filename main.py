import gc
from torch.utils.data import Dataset
import os, argparse, pickle, sys, time
import torch
from utils.metrics import *
import torch.nn.functional as F
import logging
import torchvision.models as models
import torch.nn as nn
import pandas as pd
from datetime import datetime
from scipy.special import softmax


class DSDataset(Dataset):
    def __init__(self, lead_data, ds_label, args_):
        """
        @param lead_data: 12-lead ECG array with shape of (# of ECGs, 5000, 12, 1)
        @param ds_label: diastolic function grade label array
        @param args_: arguments
        """
        self.data = np.moveaxis(lead_data, -1, 1)  # i.e., (# of ECGs, 5000, 12, 1) --> (# of ECGs, 1, 5000, 12)
        if args_.ecg == 'rhythm':
            self.data = np.reshape(self.data, (-1, self.data.shape[1], 1000, self.data.shape[3]))
            self.label = np.repeat(ds_label, 5, axis=0)
        else:  # median
            self.label = ds_label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


def load_saved_weight(init_model, weight_path):
    """
    This model is for loading weights.
    Parameters
    ----------
    init_model : initialized torch model
    weight_path : weight path for loading the trained model

    Returns loaded torch model
    -------
    """
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    init_model.load_state_dict(state_dict['model_state_dict'])

    return init_model


def train(loader_tr, device_, model_, args_):
    cur_dt = datetime.now()
    date_time = cur_dt.strftime("%m%d%Y_%H%M%S")

    if args_.result_dir is None:
        args_.result_dir = "./results"
        if not os.path.exists(args_.result_dir):
            os.makedirs(args_.result_dir)

    weight_path = f"{args_.result_dir}/{args_.arch}_{args_.num_classes}_{args_.opt}_bs{args_.batch_size}_{args_.lr}_ep{args_.ep}"

    if args_.num_leads == 1:
        weight_path += '_singleLead'
        weight_path += f'_lead{args_.specific_lead}only'
    if args_.ecg != 'rhythm':
        weight_path += '_median'
    if args_.opt != 'adam':
        weight_path += f'_{args_.opt}'

    weight_path += f'_{date_time}'
    os.makedirs(weight_path)

    # Creating logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=weight_path + '/training.log',
                        filemode='w')
    logger.info('#### Start Training ####')

    if args_.opt == 'adam':
        optimizer = torch.optim.Adam(model_.parameters(), lr=args_.lr)
    elif args_.opt == 'sgd':
        optimizer = torch.optim.SGD(model_.parameters(), lr=args_.lr)
    elif args_.opt == 'adamw':
        optimizer = torch.optim.AdamW(model_.parameters(), lr=args_.lr)

    init_time = time.time()

    criterion = torch.nn.BCEWithLogitsLoss()

    print('Start to train...')

    for ep in range(args_.ep):
        start_time = time.time()
        train_loss = 0.0
        probs = []
        target_labels = []

        for i, (data, labels) in enumerate(loader_tr):
            model_.train()
            if args_.num_classes != 5:
                labels = labels.type(torch.LongTensor)
            else:
                labels = labels.type(torch.float32)
            data, labels = data.to(device_, dtype=torch.float), labels.to(device_)
            optimizer.zero_grad()
            outputs = model_(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            probs.append(outputs.detach().cpu().numpy())
            if args_.num_classes != 5:
                target_labels.append(
                    F.one_hot(labels.to(torch.int64), num_classes=args_.num_classes).detach().cpu().numpy())
            else:
                target_labels.append(labels.to(torch.int64).detach().cpu().numpy())

        probs = np.vstack(probs)
        target_labels = np.vstack(target_labels)
        train_metric_value = get_auroc_for_multi_class(probs, target_labels)

        state = dict(epoch=ep + 1, model=model_.state_dict(), optimizer=optimizer.state_dict())
        torch.save(state, weight_path + f'/ep{ep}.pth')

        train_loss /= len(loader_tr)
        end_time = time.time()

        logger.info(
            'Epoch: [{}/{}], Step: [{}/{}], Train {}: {}, Train Loss: {}, Time: {:.2f}s'.format(
                ep + 1, args_.ep, i + 1, len(loader_tr),
                "AUROC", train_metric_value,
                train_loss, end_time - start_time))
        sys.stdout.flush()
        print(f"epoch {ep + 1} is finished!")
    state = dict(epoch=ep + 1, model=model_.state_dict(), optimizer=optimizer.state_dict())
    torch.save(state, weight_path + f'/ep{ep}.pth')
    logger.info('#### End Training ####')
    logger.info(f'Total training time: {(time.time() - init_time) / 3600:.2f} hours')


def val(loader_val, device_, model_, args_):
    weight_list = [int(i.split('ep')[1].split('.')[0]) for i in list(os.listdir(args_.saved_weight_dir))
                   if ('ep' in i) and ('pth' in i)]
    results = {}
    criterion = torch.nn.CrossEntropyLoss()

    if args_.result_dir is None:
        result_dir_to_save = args_.saved_weight_dir
    else:
        if not os.path.exists(args_.result_dir):
            os.makedirs(args_.result_dir)
        result_dir_to_save = args_.result_dir

    print('Going to validate...')

    for ep_i in range(max(weight_list) + 1):
        ckpt = torch.load(args_.saved_weight_dir + f'/ep{ep_i}.pth', map_location='cpu')
        try:
            model_.load_state_dict(ckpt['model'])
        except:
            model_.load_state_dict(ckpt)

        model_ = torch.nn.DataParallel(model_)
        model_.eval()

        results[ep_i] = {
            'target': np.zeros(len(loader_val.dataset)),
            'pred': np.zeros((len(loader_val.dataset), args_.num_classes)),
            'pred_label': np.zeros(len(loader_val.dataset))
        }
        print(f'Trained weight at epoch {ep_i} is loaded.')

        for bi, (inputs, targets) in enumerate(loader_val):
            targets = targets.type(torch.LongTensor)
            inputs_batch, targets_batch = inputs.to(device_, dtype=torch.float), targets.to(device_)
            outputs = model_(inputs_batch)
            results[ep_i]['target'][
            (bi * args_.batch_size):((bi + 1) * args_.batch_size)] = targets_batch.detach().cpu().numpy()
            results[ep_i]['pred'][
            (bi * args_.batch_size):((bi + 1) * args_.batch_size)] = outputs.detach().cpu().numpy()
            results[ep_i]['pred_label'][(bi * args_.batch_size):((bi + 1) * args_.batch_size)] = torch.argmax(outputs,
                                                                                                              dim=1).detach().cpu().numpy()

        results[ep_i]['loss'] = criterion(torch.Tensor(results[ep_i]['pred']),
                                          torch.Tensor(results[ep_i]['target']).type(torch.LongTensor)).item()
        if args_.num_classes == 2:
            fpr, tpr, _ = roc_curve(results[ep_i]['target'], softmax(results[ep_i]['pred'], axis=1)[:, 1], pos_label=1)
            results[ep_i]['auc'] = auc(fpr, tpr)
        print(f'Validation using trained weight at epoch {ep_i} is finished.')

    cur_dt = datetime.now()
    date_time = cur_dt.strftime("%m%d%Y_%H%M%S")

    result_path = f'{result_dir_to_save}/{args_.mode}_{date_time}.pkl'

    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
        print(f'{result_path} is saved!')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', default='resnet18', type=str, help='Model architecture.')
    parser.add_argument('-m', '--mode', default='training', type=str, help='training/validation/test')
    parser.add_argument('-nl', '--num_leads', default=12, type=int, help='number of leads for ECG')
    parser.add_argument('-sl', '--specific_lead', default=None, type=int, help='specific lead for single-lead ECG')
    parser.add_argument('-lr', '--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-e', '--ep', default=30, type=int, help='training epoch')
    parser.add_argument('-b', '--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('-o', '--opt', default='adam', type=str, help='optimizer; adam/sgd')
    parser.add_argument('-ecg', '--ecg', default='rhythm', type=str, help='ecg type; rhythm/median')
    parser.add_argument('-dp', '--data_path', default=None, type=str, help='data path')
    parser.add_argument('-sp', '--split_path', default=None, type=str, help='split path')
    parser.add_argument('-lp', '--label_path', default=None, type=str, help='label path')
    parser.add_argument('-rd', '--result_dir', default=None, type=str, help='result directory; default: ./results')

    parser.add_argument('--weight_sampling', action='store_true', help='If true, weighted sampling is performed.')
    args = parser.parse_args()

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    data_split = pd.read_csv(args.split_path)

    data_path = args.data_path
    label_path = args.label_path

    raw_data = np.load(data_path)
    if (args.mode == 'training') or (label_path is not None):
        raw_label = np.load(label_path)
    else:
        raw_label = np.zeros(len(raw_data))

    if args.num_leads == 1:
        if args.specific_lead is not None:
            raw_data = raw_data[:, :, (args.specific_lead - 1):args.specific_lead, :]
        else:
            assert False, "Please use an argument --specific_lead to train/validate a single lead model"

    if args.arch == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, 4)
    elif args.arch == 'resnet50':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, 4)
    elif args.arch == 'resnet101':
        model = models.resnet101(pretrained=False)
        model.fc = nn.Linear(512, 4)
    elif args.arch == 'wide-resnet50-2':
        model = models.wide_resnet50_2(pretrained=False)
        model.fc = nn.Linear(2048, 4)
    elif args.arch == 'resnext50-32x4d':
        model = models.resnext50_32x4d(pretrained=False, num_classes=4)
        model.fc = nn.Linear(2048, 4)
    else:
        assert False, "please use one of possible architectures (resnet18, resnet50, resnet101, wide-resnet50-2, resnext50-32x4d)"

    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.to(device)

    if args.mode == 'training':
        print('Training...')
        train_data = raw_data[(data_split['split'] == 'training').values.astype(bool)]
        train_label = raw_label[(data_split['split'] == 'training').values.astype(bool)]
        val_data = raw_data[(data_split['split'] == 'validation').values.astype(bool)]
        val_label = raw_label[(data_split['split'] == 'validation').values.astype(bool)]

        train_dataset = DSDataset(train_data, train_label, args)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   pin_memory=True,
                                                   drop_last=True)

        val_dataset = DSDataset(val_data, val_label, args)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False)
        del raw_data
        gc.collect()

        print('Finished to load data...')

        train(train_loader, device, model, args)
    elif args.mode in ['validation', 'test']:
        print(f'Validation ({args.mode})...')
        if args.data_path is not None:
            val_data = raw_data
            val_label = raw_label
        else:
            val_data = raw_data[(data_split['split'] == args.mode).values.astype(bool)]
            val_label = raw_label[(data_split['split'] == args.mode).values.astype(bool)]

        val_dataset = DSDataset(val_data, val_label, args)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False)
        del raw_data
        gc.collect()

        print('Finished to load data...')

        val(val_loader, device, model, args)
    else:  # external validation; e.g., indeterminate pts
        print(f'Validation ({args.mode})...')
        val_data = raw_data
        val_label = np.zeros(len(val_data))  # random label because label doesn't exist

        val_dataset = DSDataset(val_data, val_label, args)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False)
        del raw_data
        gc.collect()

        print('Finished to load data...')

        val(val_loader, device, model, args)
