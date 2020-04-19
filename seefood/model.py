""" Models for reasoning about images of food """

import torch.nn as nn

from skorch import NeuralNet
from skorch.callbacks import LRScheduler
from skorch.callbacks import Checkpoint

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted

class CalorieNet(nn.Module):
    """ Predicts calories from image features """

    def __init__(self):
        super(CalorieNet, self).__init__()
        self.regressor = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(1280, 512), nn.Linear(512, 1)
        )

    def forward(self, x):
        y = self.regressor(x)
        y = y.squeeze()
        return y


class MeanRegressionModel(BaseEstimator, RegressorMixin):

    def fit(self, _, y):
        self.mean_ = y.mean()
        return self

    def predict(self, X):
        check_is_fitted(self, [])

        return np.array(X.shape[0] * [self.mean_])


def calorie_model(val_ds):

    lrscheduler = LRScheduler(policy='StepLR', step_size=7, gamma=0.1)
    checkpoint = Checkpoint(f_params='models/calorie_net.pt', monitor='valid_acc_best')

    return NeuralNet(
        CalorieNet,
        criterion = nn.MSELoss(),
        lr = 0.001,
        batch_size = 64,
        max_epochs = 25,
        optimizer = optim.SGD,
        optimizer__momentum = 0.9,
        iterator_train__shuffle=True,
        iterator_train__num_workers=4,
        iterator_valid__shuffle=True,
        iterator_valid__num_workers=4,
        train_split=predefined_split(val_ds),
        callbacks=[lrscheduler, checkpoint],
        device='cuda'
    )


def food_recognition_model():
    return None
