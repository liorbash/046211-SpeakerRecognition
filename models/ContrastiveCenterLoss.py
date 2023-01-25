# Credit:
# https://github.com/samtwl/Deep-Learning-Contrastive-Center-Loss-Transfer-Learning-Food-Classification-/tree/master

import torch
import torch.nn as nn
from torch.autograd import Variable


class ContrastiveCenterLoss(nn.Module):
    def __init__(self, dim_hidden, num_classes, device, lambda_c=1.0):
        """
        :param dim_hidden: int. Dimension of last hidden layer's output
        :param num_classes: int
        :param device: string. 'cpu' or 'cuda:X'
        :param lambda_c: float
        """
        super(ContrastiveCenterLoss, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, dim_hidden))
        self.device = device
        self.data = {'intra_distances': [], 'inter_distances': [], 'loss': [], 'batch_size': []}

    def forward(self, y, hidden):
        """
        :param y: torch.Tensor. The true labels
        :param hidden: torch.Tensor. The output of the last hidden layer
        :return: loss
        """
        batch_size = hidden.size()[0]
        expanded_centers = self.centers.expand(batch_size, -1, -1)
        expanded_centers = expanded_centers.to(self.device)
        hidden = hidden.view(hidden.size(0), -1)
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0)
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1)
        distances_same = distance_centers.gather(1, y.unsqueeze(1))
        intra_distances = distances_same.sum()
        inter_distances = distance_centers.sum().sub(intra_distances)
        epsilon = 1e-6
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances / \
               (inter_distances + epsilon) / 0.1
        # save data
        self.data['intra_distances'].append(intra_distances.data.item())
        self.data['inter_distances'].append(inter_distances.data.item())
        self.data['loss'].append(loss.data.item())
        self.data['batch_size'].append(batch_size)
        return loss