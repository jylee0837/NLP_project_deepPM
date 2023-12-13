import torch
import torch.nn as nn


def mse_loss(output, target):
    loss_fn = nn.L1Loss(reduction='none')
    loss = loss_fn(output, target) / (target + 1e-3) 
    loss = torch.mean(loss)

    return loss

