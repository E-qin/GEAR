import torch.nn.functional as F
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy(pred, label):
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(pred, label)
    return loss