import argparse
import pickle
import glob
import sys
import os

from torch.utils.data import ConcatDataset
import pandas as pd
import torch

from model.lob2vec import (
    DeepLobPreText,
    TransLobPreText,
    DeepLobPred,
    TransLobPred,
)
from utils.optimizers import LARS, LARSWrapper
from utils import trainer, losses, make_batches
from dataset import dataset

group_columns = ['STOCK', 'DAY']


def arg_parse():
    args = argparse.ArgumentParser()
    args.add_argument(
        '-md',
        '--mode',
        default='pretext',
        type=str,
        help='mode: pretext, downstream',
    )
    args.add_argument(
        '-dp', '--data-path', default='./data/', type=str, help='path to data'
    )
    args.add_argument('-e', '--epochs', default=100, type=int, help='epochs')
    args.add_argument(
        '-d', '--device', default='cuda:0', type=str, help='device'
    )
    args.add_argument(
        '-b', '--batch-size', default=4096, type=int, help='batch size'
    )
    args.add_argument(
        '-m',
        '--model',
        default='translob',
        type=str,
        help='model to train: deeplob or translob',
    )
    args.add_argument(
        '-l',
        '--loss',
        default='vicandsupcon',
        type=str,
        help='loss scheme: supcon, vic, vicsupcon, vicandsupcon, supconmixup',
    )
    args.add_argument(
        '-lw',
        '--lr-weight',
        default=1e-3,
        type=float,
        help='learning rate for weights',
    )
    args.add_argument(
        '-lb',
        '--lr-bias',
        default=1e-4,
        type=float,
        help='learning rate for biases',
    )
    args.add_argument(
        '-ld',
        '--lr-downstream',
        default=1e-4,
        type=float,
        help='learning rate for downstream',
    )
    args.add_argument(
        '-mp',
        '--model-path',
        default='./ckpts/best_val_translob_model_pytorch',
        type=str,
        help='path to checkpoints',
    )

    return args.parse_args()


def load_data(args):
    columns = build_columns()
    df = pd.DataFrame()
    for f in glob.glob(args.data_path + '*.csv'):
        df_temp = pd.read_csv(f, index_col=0)[columns]
        df = df.append(df_temp)

    train_df = df[df.DAY < 8]
    val_df = df[df.DAY > 7]

    train_lobs = [
        v.drop(group_columns, axis=1)
        for _, v in train_df.groupby(['DAY', 'STOCK'])
    ]
    val_lobs = [
        v.drop(group_columns, axis=1)
        for _, v in val_df.groupby(['DAY', 'STOCK'])
    ]

    return train_lobs, val_lobs


def build_dataset(args, train_lobs, val_lobs):
    batch_size = args.batch_size
    if args.mode == 'pretext':
        if args.model == 'deeplob':
            datasetclass = getattr(dataset, 'DeepLobDataset')
        else:
            datasetclass = getattr(dataset, 'TransLobDataset')
    else:
        if args.model == 'deeplob':
            datasetclass = getattr(dataset, 'DownstreamDeepLobDataset')
        else:
            datasetclass = getattr(dataset, 'DownstreamTransLobDataset')
    train_dataset = ConcatDataset(
        [datasetclass(d, k=4, num_classes=3, T=100) for d in train_lobs]
    )
    val_dataset = ConcatDataset(
        [datasetclass(d, k=4, num_classes=3, T=100) for d in val_lobs]
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader


def build_columns():
    columns = []
    for i in range(10):
        columns.append('PRICE_ASK_' + str(i))
        columns.append('VOLUME_ASK_' + str(i))
        columns.append('PRICE_BID_' + str(i))
        columns.append('VOLUME_BID_' + str(i))
    for i in [1, 2, 3, 5, 10]:
        columns.append('LABEL_' + str(i) + 'TICK')
    columns = group_columns + columns

    return columns


def pretrain(args):
    device = torch.device(args.device)
    criterions = {
        'supcon': 'SupConLoss',
        'vic': 'VICLoss',
        'vicsupcon': 'VICSupConLoss',
        'vicandsupcon': 'VICandSupConMixupLoss',
        'supconmixup': 'SupConMixUpLoss',
    }

    tran_lobs, val_lobs = load_data(args)
    train_loader, val_loader = build_dataset(args, tran_lobs, val_lobs)
    if 'supcon' in args.loss:
        norm = True
        # norm = False
    else:
        norm = False
    if args.model == 'deeplob':
        pretextmodel = DeepLobPreText(norm=norm).to(device)
    else:
        pretextmodel = TransLobPreText(norm=norm).to(device)

    param_weights = []
    param_biases = []
    for param in pretextmodel.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    optimizer = LARS(parameters, 0)  # WORKS
    # optimizer = LARS(pretextmodel.parameters(), 100)
    # optimizer = torch.optim.RMSprop(pretextmodel.parameters(), 1e-5)
    criterion = getattr(losses, criterions.get(args.loss))()
    train_fn = getattr(trainer, args.mode + '_train_function')
    make_batches_fn = None
    if 'sup' not in args.loss:
        # make_batches_fn = make_batches.bt_three_way
        make_batches_fn = make_batches.bt_aug1_vs_aug2
    train_losses, val_losses = train_fn(
        args,
        pretextmodel,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        make_batches_fn,
    )

    return train_losses, val_losses


def train(args):
    device = torch.device(args.device)

    tran_lobs, val_lobs = load_data(args)
    train_loader, val_loader = build_dataset(args, tran_lobs, val_lobs)
    if 'supcon' in args.loss:
        norm = True
    else:
        norm = False
    enc = torch.load(args.model_path).enc
    if args.model == 'deeplob':
        predmodel = DeepLobPred(enc=enc, norm=norm).to(device)
    else:
        predmodel = TransLobPred(enc=enc, norm=norm).to(device)
    optimizer = torch.optim.Adam(predmodel.parameters(), args.lr_downstream)
    criterion = torch.nn.CrossEntropyLoss()
    train_fn = getattr(trainer, args.mode + '_train_function')
    train_losses, val_losses = train_fn(
        args,
        predmodel,
        criterion,
        optimizer,
        train_loader,
        val_loader,
    )

    return train_losses, val_losses


if __name__ == '__main__':
    sys.path.append(
        os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    )
    args = arg_parse()
    if args.mode == 'pretext':
        train_losses, val_losses = pretrain(args)
    else:
        train_losses, val_losses = train(args)

    with open('./losses.pkl', 'wb') as f:
        pickle.dump((train_losses, val_losses), f)