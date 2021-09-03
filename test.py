from model.lob2vec import (
    DeepLobPreText,
    TransLobPreText,
    DeepLobPred,
    TransLobPred,
)
import torch
import numpy as np

if __name__ == '__main__':
    model = torch.load(
        '/home/user/Documents/git/lob2vec/ckpts/best_val_translob_model_pytorch'
    )
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]

    print(np.mean(parameters[0]['params']))
    print(np.mean(parameters[1]['params']))