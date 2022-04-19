import torch
import torch.nn as nn
from torchvision import models

class ResNetEmbed(nn.Module):
    def __init__(self, leaf_nodes, total_nodes):
        super(ResNetEmbed, self).__init__()
        # Define the share part
        self.trunk = self._make_trunk()
        # Define the hierarchy root and leaf branch
        self.fc_total = nn.Sequential(nn.Linear(2048, 256),
                                      nn.ReLU(),
                                      nn.Dropout(0.5),
                                      nn.Linear(256, total_nodes),    # raw classification scores
                                      nn.Sigmoid()  # sigmoid to ensure the outputs are in the range(0,1)
                                      )
        # Define the softmax leaf branch
        self.fc_leaf = nn.Sequential(nn.Linear(2048, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(256, leaf_nodes),
                                     nn.LogSoftmax(dim=1)
                                     )

    def _make_trunk(self):
        resnet50 = models.resnet50(pretrained=False)
        resnet50.load_state_dict(torch.load('./pre-trained/resnet50-19c8e357.pth'))
        trunk = nn.Sequential(*(list(resnet50.children())[:-1]))
        return trunk

    def forward(self, x):
        f_share = self.trunk(x)
        f_share = f_share.view(f_share.size(0), -1)
        fc_total = self.fc_total(f_share)
        fc_leaf = self.fc_leaf(f_share)
        return fc_leaf, fc_total

