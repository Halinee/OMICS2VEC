from adabelief_pytorch import AdaBelief
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs, targets)

        pt = th.exp(-ce_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduce:
            return th.mean(F_loss)
        else:
            return F_loss


OPTIMIZER_FACTORY = {
    "adabelief": AdaBelief,
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "adam": optim.Adam,
    "sparseadam": optim.SparseAdam,
    "adamax": optim.Adamax,
    "rmsprop": optim.RMSprop,
    "rprop": optim.Rprop,
    "asgd": optim.ASGD,
    "lbfgs": optim.LBFGS,
    "sgd": optim.SGD,
}

ACTIVATION_FN_FACTORY = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "tanh": nn.Tanh(),
    "lrelu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "celu": nn.CELU(),
}

LOSS_FN_FACTORY = {
    "mse": nn.MSELoss(),
    "mae": nn.L1Loss(),
    "bce": nn.BCELoss(),
    "focal": FocalLoss(),
}
