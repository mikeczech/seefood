""" Models for reasoning about images of food """

import torch.nn as nn

class CalorieNet(nn.Module):
    """ Predicts calories given an image displaying food """

    def __init__(self, model):
        super(CalorieNet, self).__init__()
        self.features = model.features
        self.regressor = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(1280, 512), nn.Linear(512, 1)
        )
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        y = self.regressor(x)
        y = y.squeeze()
        return y
