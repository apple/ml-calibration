#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import numpy as np
import sklearn
from sklearn.base import BaseEstimator


class KernelSmoother(BaseEstimator):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def fit(self, X, y):
        # X: (n_samples, n_features)
        # y: (n_samples,)
        if X.shape[1] != 1:
            raise ValueError("Data should be 1-dimensional.")
        self.X_ = X.astype(np.double)
        self.y_ = y.astype(np.double)
        return self

    def predict(self, x):
        if x.shape[1] != 1:
            raise ValueError("Data should be 1-dimensional.")
        ys, _ = self.kernel.smooth(self.X_.reshape(-1), self.y_, x.reshape(-1))
        # XXX We can compute convolution in the fit phase and just interpolation here.
        return ys

    predict_proba = predict

class Binning(BaseEstimator):
    def __init__(self, bins_cnt=30, range=(0, 1)):
        super().__init__()
        self.bins_cnt = bins_cnt
        self.range = range

    def _get_bucket_indices(self, x):
        return (
            ((x - self.range[0]) / (self.range[1] - self.range[0]) * self.bins_cnt)
            .astype(int)
            .clip(0, self.bins_cnt - 1)
        )

    def fit(self, data, labels):
        if data.shape[1] != 1:
            raise ValueError("Data should be 1-dimensional.")

        bucket_indices = self._get_bucket_indices(data[:, 0])
        buckets = np.zeros(self.bins_cnt)
        bucket_sizes = np.zeros(self.bins_cnt, dtype=int)
        for i, y in zip(bucket_indices, labels):
            buckets[i] += y
            bucket_sizes[i] += 1

        self.bucket_predictions_ = np.array(
            [b / bs if bs > 0 else 0 for (b, bs) in zip(buckets, bucket_sizes)]
        )
        self.bucket_sizes_ = bucket_sizes
        return self

    def predict(self, data):
        if data.shape[1] != 1:
            raise ValueError("Data should be 1-dimensional.")
        bucket_indices = self._get_bucket_indices(data[:, 0])
        return self.bucket_predictions_[bucket_indices]

    predict_proba = predict


class CenteredRegressor(BaseEstimator):
    """
    Estimator that transforms regression targets (x, y) --> (x, y-x), calls a base-regressor, and then inverts the transform.
    """

    def __init__(self, regressor):
        super().__init__()
        self.regressor = regressor

    def fit(self, X, y, **fit_params):
        # X: (n_samples, 1)
        # y: (n_samples,)
        self.regressor_ = sklearn.base.clone(self.regressor)
        self.regressor_.fit(X, (y - X.reshape(-1)), **fit_params)
        return self

    def predict(self, x, **predict_params):
        pred = self.regressor_.predict(x, **predict_params).reshape(-1)
        pred += x.reshape(-1)
        return pred


class RecalibratedEstimator(BaseEstimator):
    def __init__(self, estimator, recalibration_regressor):
        self.estimator = estimator
        self.recalibration_regressor = recalibration_regressor

    def fit(self, X, y, **fit_params):
        self.estimator_ = sklearn.base.clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        y_pred = self.estimator_.predict_proba(X)
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        y_pred = y_pred.reshape((len(y_pred), 1))
        self.recalibration_regressor_ = sklearn.base.clone(self.recalibration_regressor)
        self.recalibration_regressor_.fit(y_pred, y)

    def predict(self, x, **predict_params):
        pred = self.estimator_.predict_proba(x, **predict_params)
        if len(pred.shape) == 2 and pred.shape[1] == 2:
            pred = pred[:, 1]
        pred = pred.reshape((len(pred), 1))
        return self.recalibration_regressor_.predict(pred)

    ## Commented out to avoid sklearn version incompatibility
    # def _sk_visual_block_(self):
    #     from sklearn.utils._repr_html.estimator import _VisualBlock
    #     estimation_block = _VisualBlock(
    #         "parallel", [self.estimator], names=["estimator"], dash_wrapped=False
    #     )
    #     recalibration_block = _VisualBlock(
    #         "parallel",
    #         [self.recalibration_regressor],
    #         names=[
    #             f"recalibration_regressor: {self.recalibration_regressor.__class__.__name__}"
    #         ],
    #         dash_wrapped=False,
    #     )
    #     return _VisualBlock(
    #         "serial", [estimation_block, recalibration_block], dash_wrapped=False
    #     )

    predict_proba = predict