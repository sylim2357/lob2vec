import glob

from torch.utils.data import ConcatDataset, Dataset
import pandas as pd
import torch

from dataset.dataset import DeepLobDataset, TransLobDataset
from train.trainer import pretext_train_function
from model.lob2vec import TransLobPreText
from train.optimizers import LARS
from train.losses import VICLoss

group_columns = ['STOCK', 'DAY']


def load_data(args):
    columns = build_columns()
    df = pd.DataFrame()
    for f in glob.glob(args.data_path):
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
    if args.model == 'deeplob':
        datasetclass = DeepLobDataset
    else:
        datasetclass = TransLobDataset
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


def train(args):
    device = torch.device(args.device)
    tran_lobs, val_lobs = load_data(args)
    train_loader, val_loader = build_dataset(args, tran_lobs, val_lobs)
    pretextmodel = TransLobPreText().to(device)

    param_weights = []
    param_biases = []
    for param in pretextmodel.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    optimizer = LARS(
        parameters, 0, weight_decay_filter=True, lars_adaptation_filter=True
    )  # WORKS
    vic_criterion = VICLoss()

    pretext_train_function(
        args,
        pretextmodel,
        vic_criterion,
        optimizer,
        train_loader,
        val_loader,
    )
