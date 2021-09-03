import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
import math


def pretext_train_function(
    args,
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    make_batches=None,
):
    if 'sup' in args.loss:
        assert make_batches is None

    device = torch.device(args.device)
    train_losses = np.zeros(args.epochs)
    test_losses = np.zeros(args.epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in tqdm(range(args.epochs)):

        model.train()
        t0 = datetime.now()
        train_loss = []
        for idx, (inputs, targets) in enumerate(train_loader):
            inp = inputs.to(device, dtype=torch.float)  # B x views x channels
            batch_size = inp.size(0)
            n_views = inp.size(1)
            adjust_learning_rate(
                optimizer,
                train_loader,
                idx,
                args.epochs,
                inp.size(0),
                args.lr_weight,
                args.lr_bias,
            )

            optimizer.zero_grad()
            x = inp.view(batch_size * n_views, *(inp.size()[2:]))
            z = model(x).reshape(batch_size, n_views, -1)

            if 'sup' in args.loss:
                targets = targets.to(device, dtype=torch.int64)
                assert z.size(0) == targets.size(0)
                if idx % 100 == 0:
                    printbool = True
                else:
                    printbool = False
                loss = criterion(features=z, labels=targets, print_c=printbool)
                # loss = criterion(features=z, labels=targets)

            else:
                assert (
                    z.size(1) > 1
                ), 'unsupervised setting should have more than 1 view'
                z1, z2 = make_batches(z)
                if idx % 100 == 0:
                    printbool = True
                else:
                    printbool = False
                loss = criterion(z1, z2, print_c=printbool)

            loss.backward()
            optimizer.step()

            if idx % 100 == 0:
                print(f'{idx} update: {loss.item()}')
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss)  # a little misleading

        model.eval()
        test_loss = []
        for t_idx, (inputs, targets) in enumerate(test_loader):
            inp = inputs.to(device, dtype=torch.float)  # B x views x channels
            batch_size = inp.size(0)
            n_views = inp.size(1)

            x = inp.view(batch_size * n_views, *(inp.size()[2:]))
            z = model(x).reshape(batch_size, n_views, -1)

            if 'sup' in args.loss:
                targets = targets.to(device, dtype=torch.int64)
                assert z.size(0) == targets.size(0)
                if t_idx % 100 == 0:
                    printbool = True
                    print('test!')
                else:
                    printbool = False
                loss = criterion(features=z, labels=targets, print_c=printbool)
                # loss = criterion(features=z, labels=targets)

            else:
                z1, z2 = make_batches(z)
                if t_idx % 100 == 0:
                    printbool = True
                    print('test!')
                else:
                    printbool = False
                loss = criterion(z1, z2, print_c=printbool)

            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        if test_loss < best_test_loss:
            torch.save(model, args.model_path)
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(
            f'Epoch {it+1}/{args.epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}'
        )

    return train_losses, test_losses


def adjust_learning_rate(
    optimizer,
    loader,
    step,
    epochs,
    batch_size,
    learning_rate_weights,
    learning_rate_biases,
):
    max_steps = epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * learning_rate_biases


def downstream_train_function(
    args,
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
):
    device = torch.device(args.device)
    train_losses = np.zeros(args.epochs)
    test_losses = np.zeros(args.epochs)
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in tqdm(range(args.epochs)):

        model.train()
        t0 = datetime.now()
        train_loss = []
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = (
                inputs.to(device, dtype=torch.float),
                targets.to(device, dtype=torch.int64),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if idx % 100 == 0:
                print(f'{idx} update: {loss.item()}')
        train_loss = np.mean(train_loss)  # a little misleading

        model.eval()
        test_loss = []
        n_correct = 0.0
        n_total = 0.0
        for inputs, targets in test_loader:
            inputs, targets = (
                inputs.to(device, dtype=torch.float),
                targets.to(device, dtype=torch.int64),
            )
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predictions = torch.max(outputs, 1)

            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]

            test_loss.append(loss.item())

        test_loss = np.mean(test_loss)
        test_acc = n_correct / n_total
        print(f"Test acc: {test_acc:.4f}")

        train_losses[it] = train_loss
        test_losses[it] = test_loss

        if test_loss < best_test_loss:
            # torch.save(model, './best_val_model_pytorch')
            best_test_loss = test_loss
            best_test_epoch = it
            # print('model saved')

        dt = datetime.now() - t0
        print(
            f'Epoch {it+1}/{args.epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}'
        )

    return train_losses, test_losses