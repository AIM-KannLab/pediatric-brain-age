import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.densenet import densenet121

from timm.models import create_model

import numpy as np
#diffmc
class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        gamma = gamma.unsqueeze(1)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, config, guidance=False):
        super(ConditionalModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1
        data_dim = config.model.data_dim
        y_dim = 1 #config.data.num_classes
        arch = config.model.arch
        feature_dim = config.model.feature_dim
        hidden_dim = config.model.hidden_dim
        self.guidance = guidance
        # encoder for x
        self.encoder_x = ResNetEncoder(arch=arch, feature_dim=feature_dim)
        # batch norm layer
        self.norm = nn.BatchNorm1d(feature_dim)

        # Unet
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim * 2, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)
        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(self, x, y, t, yhat=None):
        x = self.encoder_x(x)
        x = self.norm(x)
        if self.guidance:
            #for yh in yhat:
            y = torch.cat([y, yhat], dim=-1)
        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.relu(y)
        y = x * y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.relu(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.relu(y)
        y = self.lin4(y)

        return y



# ResNet 18 or 50 as image encoder
class ResNetEncoder(nn.Module):
    def __init__(self, arch='resnet18', feature_dim=128):
        super(ResNetEncoder, self).__init__()

        self.f = []
        #print(arch)
        if arch == 'resnet50':
            backbone = resnet50()
            self.featdim = backbone.fc.weight.shape[1]
        elif arch == 'resnet18':
            backbone = resnet18()
            self.featdim = backbone.fc.weight.shape[1]
        elif arch == 'densenet121':
            backbone = densenet121(pretrained=True)
            self.featdim = backbone.classifier.weight.shape[1]
        elif arch == 'vit':
            backbone = create_model('pvt_v2_b2',
            pretrained=True,
            num_classes=4,
            drop_rate=0,
            drop_path_rate=0.1,
            drop_block_rate=None,
            )
            backbone.head = nn.Sequential()
            self.featdim = 512

        for name, module in backbone.named_children():
            #if not isinstance(module, nn.Linear):
            #    self.f.append(module)
            if name != 'fc':
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        
        #print(self.featdim)
        self.g = nn.Linear(self.featdim, feature_dim)
        #self.z = nn.Linear(feature_dim, 4)

    def forward_feature(self, x):
        feature = self.f(x)
        #x = x.mean(dim=1)

        feature = torch.flatten(feature, start_dim=1)
        feature = self.g(feature)

        return feature

    def forward(self, x):
        feature = self.forward_feature(x)
        return feature

# early stopping scheme for hyperparameter tuning
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): Number of steps to wait after average improvement is below certain threshold.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement;
                           shall be a small positive value.
                           Default: 0
            best_score: value of the best metric on the validation set.
            best_epoch: epoch with the best metric on the validation set.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, val_cost, epoch, verbose=False):

        score = val_cost

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch + 1
        elif score > self.best_score - self.delta:
            self.counter += 1
            if verbose:
                print("EarlyStopping counter: {} out of {}...".format(
                    self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.counter = 0