import numpy as np
from datetime import datetime
import tqdm
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
    if args['supervised']:
        assert make_batches is None

    train_losses = np.zeros(args['epochs'])
    test_losses = np.zeros(args['epochs'])
    best_test_loss = np.inf
    best_test_epoch = 0

    for it in tqdm(range(args['epochs'])):

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
                args['epochs'],
                inp.size(0),
                args['learning_rate_weights'],
                args['learning_rate_biases'],
            )

            optimizer.zero_grad()
            x = inp.view(batch_size * n_views, *(inp.size()[2:]))
            z = model(x).reshape(batch_size, n_views, -1)

            if args['supervised']:
                targets = targets.to(device, dtype=torch.int64)
                assert z.size(0) == targets.size(0)
                if idx % 100 == 0:
                    printbool = True
                else:
                    printbool = False
                # loss = criterion(features=z, labels=targets, print_c=printbool)
                loss = criterion(features=z, labels=targets)

            else:
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
        for inputs, targets in test_loader:
            inp = inputs.to(device, dtype=torch.float)  # B x views x channels
            batch_size = inp.size(0)
            n_views = inp.size(1)

            x = inp.view(batch_size * n_views, *(inp.size()[2:]))
            z = model(x).reshape(batch_size, n_views, -1)

            if args['supervised']:
                targets = targets.to(device, dtype=torch.int64)
                assert z.size(0) == targets.size(0)
                if idx % 100 == 0:
                    printbool = True
                    print('test!')
                else:
                    printbool = False
                # loss = criterion(features=z, labels=targets, print_c=printbool)
                loss = criterion(features=z, labels=targets)

            else:
                z1, z2 = make_batches(z)
                if idx % 100 == 0:
                    printbool = True
                else:
                    printbool = False
                loss = criterion(z1, z2, print_c=printbool)

            test_loss.append(loss.item())
        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[it] = train_loss
        test_losses[it] = test_loss

        if test_loss < best_test_loss:
            torch.save(model, './best_val_translob_model_pytorch')
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(
            f'Epoch {it+1}/{args["epochs"]}, Train Loss: {train_loss:.4f}, \
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
