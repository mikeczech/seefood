""" Models for reasoning about images of food """

import torch.nn as nn

from skorch import NeuralNet
from skorch.callbacks import LRScheduler
from skorch.callbacks import Checkpoint

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.base import RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.mixture import GaussianMixture

from sklego.pipeline import DebugPipeline

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import logging

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

class PrintPreprocessStats(BaseEstimator, TransformerMixin):

    def __init__(self, pca):
        self._pca = pca

    def fit(self, X, y = None):
        pca_explained_var = self._pca.explained_variance_ratio_
        logging.info(f"[PCA] explained variance mean: {pca_explained_var.mean()}, std: {pca_explained_var.std()}")
        return self

    def transform(self, X, y = None):
        return X


def food_recognition_model(**params):
    pca = PCA(512, whiten=True)
    return DebugPipeline(steps=[
        ("scale", StandardScaler()),
        ("pca", pca),
        ("print_preprocess_stats", PrintPreprocessStats(pca)),
        ("model", GaussianMixture(n_components=4, covariance_type='full', max_iter=int(1e7)))
    ], log_callback='default').set_params(**params)
