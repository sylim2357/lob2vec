import numpy as np
from datetime import datetime
from tqdm import tqdm
import torch
import math
import gc
import tensorboard_logger as tb_logger
from utils.losses import SupConMixUpLoss, VICandSupConLoss, VICLoss, VICSupConLoss
from model.lob2vec import (
    DeepLobPreText,
    DeepLobPred,
)
from sklearn.metrics import confusion_matrix

def pretext_train_function(
    args,
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    # make_batches=None,
):
    logger = tb_logger.Logger(logdir='D:/nathan/lob2vec/board/' + args.model + '_' +str(args.k)+'_'+str(args.batch_size), flush_secs=2)

    # if 'sup' in args.loss:
    #     assert make_batches is None

    device = torch.device(args.device)
    train_losses = np.zeros(args.epochs)
    test_losses = np.zeros(args.epochs)
    best_test_loss = np.inf
    best_test_epoch = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20)
    for it in tqdm(range(args.epochs)):

        model.train()
        t0 = datetime.now()
        train_loss = []
        iters = len(train_loader)
        for idx, ((inp1, inp2), targets) in enumerate(train_loader):
            inp1 = inp1.to(device, dtype=torch.float)  # B x views x channels
            inp2 = inp2.to(device, dtype=torch.float)  # B x views x channels
            batch_size = inp1.size(0)
            # n_views = inp.size(1)
            adjust_learning_rate(
                optimizer,
                train_loader,
                idx,
                args.epochs,
                batch_size,
                args.lr_weight,
                args.lr_bias,
            )
            optimizer.zero_grad()
            # x = inp.view(batch_size * n_views, *(inp.size()[2:]))
            z1 = model(inp1)
            z2 = model(inp2)
            # z = model(x).reshape(batch_size, n_views, -1)

            if 'sup' in args.loss:
                z = torch.stack((z1, z2), dim=1)
                targets = targets.to(device, dtype=torch.int64)
                assert z.size(0) == targets.size(0)
                if idx % 100 == 0:
                    printbool = True
                else:
                    printbool = False
                loss = criterion(features=z, labels=targets, print_c=printbool)
                # loss = criterion(features=z, labels=targets)

            else:
                # assert (
                #     z.size(1) > 1
                # ), 'unsupervised setting should have more than 1 view'
                # z1, z2 = make_batches(z)
                if idx % 100 == 0:
                    printbool = True
                else:
                    printbool = False
                loss = criterion(z1, z2, print_c=printbool)
            if printbool:
                if idx % 200:
                    print('z1')
                    print(z.squeeze())
                a = []
                for name, param in model.named_parameters():
                    a.append(param.mean().item())
                print('weight before')
                print(np.mean(a))
            loss.backward()
            optimizer.step()
            scheduler.step(it + idx / iters)
            if printbool:
                print('activation: ')
                print(z1)
                a = []
                for param in model.parameters():
                    a.append(param.mean().item())
                print('weight after')
                print(np.mean(a))

            if idx % 100 == 0:
                print(f'{idx} update: {loss.item()}')
            train_loss.append(loss.item())
        # Get train loss and test loss
        train_loss = np.mean(train_loss)  # a little misleading

        logger.log_value('loss', train_loss, it)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], it)

        gc.collect()
        model.eval()
        test_loss = []
        for t_idx, ((inp1, inp2), targets) in enumerate(test_loader):
            inp1 = inp1.to(device, dtype=torch.float)  # B x views x channels
            inp2 = inp2.to(device, dtype=torch.float)  # B x views x channels
            batch_size = inp1.size(0)
            # inp = inputs.to(device, dtype=torch.float)  # B x views x channels
            # batch_size = inp.size(0)
            # n_views = inp.size(1)

            # x = inp.view(batch_size * n_views, *(inp.size()[2:]))
            z1 = model(inp1)
            z2 = model(inp2)
            # z = model(x).reshape(batch_size, n_views, -1)

            if 'sup' in args.loss:
                z = torch.stack((z1, z2), dim=1)
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
                # z1, z2 = make_batches(z)
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

        if (test_loss < best_test_loss) or (it % 50 == 0):
            model_path = './ckpts/' + args.model + '_' + args.loss +'_k_' + str(args.k) + '_epoch_' + str(it) + '_trloss_' + str(train_loss) + '_valloss_' + str(test_loss)
            torch.save(model, model_path)
            if (test_loss < best_test_loss):
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
    warmup_steps = 100 * len(loader)
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

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 20)
    for it in tqdm(range(args.epochs)):

        model.train()
        t0 = datetime.now()
        train_loss = []
        iters = len(train_loader)
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
            scheduler.step(it + idx / iters)
            train_loss.append(loss.item())
            if idx % 100 == 0:
                print(f'{idx} update: {loss.item()}')
        train_loss = np.mean(train_loss)  # a little misleading

        model.eval()
        test_loss = []
        n_correct = 0.0
        n_total = 0.0
        tgt = []
        pred = []
        for inputs, targets in test_loader:
            inputs, targets = (
                inputs.to(device, dtype=torch.float),
                targets.to(device, dtype=torch.int64),
            )
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predictions = torch.max(outputs, 1)
            tgt += list(targets.cpu())
            pred += list(predictions.cpu())

            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]

            test_loss.append(loss.item())
        conf_matrix = confusion_matrix(tgt, pred)
        print(conf_matrix)
        test_loss = np.mean(test_loss)
        test_acc = n_correct / n_total
        print(f"Test acc: {test_acc:.4f}")

        train_losses[it] = train_loss
        test_losses[it] = test_loss

        if test_loss < best_test_loss:
            torch.save(model, './ckpts/' + args.model + '_' + args.loss +'_k_' + str(args.k) + '_downstream')
            best_test_loss = test_loss
            best_test_epoch = it
            print('model saved')

        dt = datetime.now() - t0
        print(
            f'Epoch {it+1}/{args.epochs}, Train Loss: {train_loss:.4f}, \
          Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}'
        )

    return train_losses, test_losses

def pt_train_function(
    args,
    deeplob,
    pretext_train_loader,
    train_loader,
    test_loader,
):
    device = torch.device(args.device)
    predmodel = DeepLobPred(deeplob=deeplob).to(device)
    pretextmodel = DeepLobPreText(deeplob=deeplob).to(device)
    if torch.cuda.device_count() > 1:
        gpu_id = [i for i in range(torch.cuda.device_count())]
        pretextmodel = torch.nn.DataParallel(pretextmodel)
        predmodel = torch.nn.DataParallel(predmodel)

    train_losses = np.zeros(args.epochs)
    test_losses = np.zeros(args.epochs)
    best_test_loss = np.inf
    best_test_epoch = 0
    pretext_optimizer = torch.optim.AdamW(pretextmodel.parameters(), 1e-3)
    pred_optimizer = torch.optim.AdamW(predmodel.parameters(), args.lr_downstream)
    # supconmixup = SupConMixUpLoss()
    supconmixup = VICLoss()
    celoss = torch.nn.CrossEntropyLoss()

    pt_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(pretext_optimizer, 10)
    pred_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(pred_optimizer, 10)
    for it in tqdm(range(args.epochs)):
        if it % 2 == 0:
            pretextmodel.train()
            train_loss = []
            iters = len(pretext_train_loader)
            for idx, (inputs, targets) in enumerate(pretext_train_loader):
                inp = inputs.to(device, dtype=torch.float)  # B x views x channels
                batch_size = inp.size(0)
                n_views = inp.size(1)
                pretext_optimizer.zero_grad()
                x = inp.view(batch_size * n_views, *(inp.size()[2:]))
                z = pretextmodel(x).reshape(batch_size, n_views, -1)

                # for vic
                # inp = inputs[0].unsqueeze(0).to(device, dtype=torch.float)  # B x views x channels
                # batch_size = inp.size(0)
                # n_views = inp.size(1)
                # pretext_optimizer.zero_grad()
                # x = inp.view(batch_size * n_views, *(inp.size()[2:]))
                # z = pretextmodel(x).reshape(batch_size, -1)
                #
                # aug = inputs[2].unsqueeze(0).to(device, dtype=torch.float)
                # batch_size = inp.size(0)
                # n_views = inp.size(1)
                # x_aug = aug.view(batch_size * n_views, *(aug.size()[2:]))
                # z_aug = pretextmodel(x_aug).reshape(batch_size, -1)
                # for vic

                # targets = targets.to(device, dtype=torch.int64)
                # assert z.size(0) == targets.size(0)
                # loss = supconmixup(features=z, labels=targets, print_c=idx % 100 == 0)
                loss = supconmixup(features=z, aug=z_aug, print_c=idx % 100 == 0)

                if idx % 100 == 0:
                    print(f'{idx} update: {loss.item()}')
                # loss = criterion(features=z, labels=targets)
                loss.backward()
                pretext_optimizer.step()
                pt_scheduler.step(it + idx / iters)

                if idx % 200 == 0:
                    print('activation: ')
                    print(z[:,0,::].squeeze())
                train_loss.append(loss.item())
            # Get train loss and test loss
            train_loss = np.mean(train_loss)  # a little misleading
        else:
            t0 = datetime.now()
            predmodel.train()
            train_loss = []
            iters = len(train_loader)
            for idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = (
                    inputs.to(device, dtype=torch.float),
                    targets.to(device, dtype=torch.int64),
                )
                pred_optimizer.zero_grad()
                outputs = predmodel(inputs)
                loss = celoss(outputs, targets)
                loss.backward()
                pred_optimizer.step()
                pred_scheduler.step(it + idx / iters)
                train_loss.append(loss.item())
                if idx % 100 == 0:
                    print(f'{idx} update: {loss.item()}')
            train_loss = np.mean(train_loss)  # a little misleading

            predmodel.eval()
            test_loss = []
            n_correct = 0.0
            n_total = 0.0
            for inputs, targets in test_loader:
                inputs, targets = (
                    inputs.to(device, dtype=torch.float),
                    targets.to(device, dtype=torch.int64),
                )
                outputs = predmodel(inputs)
                loss = celoss(outputs, targets)

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
                torch.save(predmodel, './ckpts/alt/' + args.model + '_' + args.loss +'_k_' + str(args.k) + '_downstream')
                torch.save(pretextmodel, './ckpts/alt/' + args.model + '_' + args.loss +'_k_' + str(args.k) + '_pretext')
                best_test_loss = test_loss
                best_test_epoch = it
                print('model saved')

            dt = datetime.now() - t0
            print(
                f'Epoch {it+1}/{args.epochs}, Train Loss: {train_loss:.4f}, \
              Validation Loss: {test_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}'
            )

    return train_losses, test_losses
