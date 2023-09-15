#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import warnings
import numpy as np
import scipy as sp
from deprecation import deprecated
from . import config


warnings.filterwarnings("default", category=DeprecationWarning,
                                   module="relplot.kernel_utils")

def kernel_cross(f, g, kernel_func):
    """
    Returns matrix K_{i, j} = k(|f_i - g_j|)
    """
    F = np.tile(np.array(f).reshape(-1, 1), (1, len(g)))
    G = np.tile(np.array(g), (len(f), 1))
    D = F - G
    K = kernel_func(np.abs(D))
    return K


def interpolate(t, y):
    """
    Evaluates linear interpolation of a function mapping i/(len[y] - 1) -> y[i] on all points t.
    """
    buckets_cnt = len(y)
    bucket_size = 1 / (buckets_cnt - 1)
    inds = (t * (buckets_cnt - 1)).astype(int).clip(0, buckets_cnt - 2)
    residual = (t - inds * bucket_size) / bucket_size
    return y[inds] * (1 - residual) + y[inds + 1] * residual


def kernel_smooth_basic(x, y, x_eval, kernel_func):
    # Kernel-smooths the values {y_i}, located at {x_i}, and evaluates at points x_eval
    # x.shape: (N,)
    # y.shape: (N,)
    t = x_eval
    Ktx = kernel_cross(t, x, kernel_func)
    dens = Ktx.dot(np.ones_like(y))
    ys = np.divide(
        Ktx.dot(y), dens, out=np.zeros_like(t), where=dens != 0
    )  # handle potential divide-by-zeros
    return ys


class BaseKernelMixin:
    def smooth(self, f, y, x_eval, eps = 0.0001):
        ys = self.apply(f, y, x_eval)
        dens = self.apply(f, np.ones_like(y), x_eval) + eps
        ys = ys / dens
        return ys, dens

    def kde(self,
            x, x_eval, debias_boundary=True, bounds=(0, 1), eps=0.00001):
        weights = np.ones_like(x) / len(x)
        density = self.apply(x, weights, x_eval) + eps
        if debias_boundary:
            # This uses an approximation (for non-Gaussian kernels)
            # kde of uniform distribution over (0, 1)
            z = np.linspace(*bounds, 1000)
            bias = self.kde(z, x_eval, debias_boundary=False) + eps
            density /= bias
        return density


class LogitGaussianKernel(BaseKernelMixin):
    def __init__(self, sigma):
        self.sigma = sigma

    def transform(self, f, x_eval, eps=0.001):
        z = x_eval.reshape(-1).astype(np.double)[1:-1]
        logit_z = np.log(z/(1-z))
        f = np.clip(f, eps, 1-eps)
        logit_f = np.log(f/(1-f))
        range_min = min(np.min(logit_f), np.min(logit_z))
        range_max = max(np.max(logit_f), np.max(logit_z))
        logit_f = (logit_f - range_min) / (range_max - range_min)
        logit_z = (logit_z - range_min) / (range_max - range_min)
        return logit_f, logit_z, 4/(range_max - range_min)

    def smooth(self, f, y, x_eval, eps=0.001):
        logit_f, logit_z, scale = self.transform(f, x_eval)
        kernel = GaussianKernel(self.sigma * scale)
        ys, dens = kernel.smooth(logit_f, y, logit_z)
        dens *= 1/(x_eval[1:-1] * (1-x_eval[1:-1]))
        dens *= len(f) * len(x_eval) / np.sum(dens)
        return np.concatenate( ([ys[0]], ys, [ys[-1]])), np.concatenate( ([dens[0]], dens, [dens[-1]]))

    def kde(self, x, x_eval):
        return self.smooth(x, np.ones_like(x), x_eval)[1]

class GaussianKernel(BaseKernelMixin):
    def __init__(self, sigma):
        self.sigma = sigma

    def kernel_ev(self, num_eval_points : int):
        t = np.linspace(0, 1, num_eval_points)
        var = self.sigma ** 2
        res = np.exp(-(t - 0.5) * (t - 0.5) / (2*var))
        res /= np.sqrt(2 * np.pi) * self.sigma
        #res *= len(t) / np.sum(res)
        return res

    def convolve(self, values, eval_points):
        ker = self.kernel_ev(eval_points)
        return np.convolve(values, ker, 'same')

    def apply(self, f, y, x_eval, eval_points = None):
        if eval_points is None:
            eval_points = max(2000, round(20 / self.sigma))
#        eval_points = 40_000
        eval_points = (eval_points // 2) + 1
        values = smooth_round_to_grid(f, y, eval_points = eval_points)
        smoothed = self.convolve(values, eval_points)
        return interpolate(x_eval, smoothed)

    def kde(self, x, x_eval):
        weights = np.ones_like(x) / len(x)
        density = self.apply(x, weights, x_eval)
        return density

class ReflectedGaussianKernel(GaussianKernel):
    def convolve(self, values, eval_points):
        ker = self.kernel_ev(eval_points)
        ext_vals = np.concatenate([np.flip(values)[:-1], values, np.flip(values)[1:]])
        return np.convolve(ext_vals, ker, "valid")[eval_points//2 : eval_points//2 + eval_points]

class TruncatedGaussianKernel(BaseKernelMixin):
    def __init__(self, sigma):
        self.sigma = sigma

    def kernel_ev(self, num_eval_points : int):
        t = np.linspace(0, 1, num_eval_points)
        var = self.sigma ** 2
        res = np.exp(-(t - 0.5) * (t - 0.5) / (2*var))
        res /= np.sqrt(2 * np.pi) * self.sigma
        return res

    def convolve(self, values, eval_points):
        ker = self.kernel_ev(eval_points)
        return np.convolve(values, ker, 'same')

    def apply(self, f, y, x_eval, eval_points = None):
        if eval_points is None:
            eval_points = max(2000, round(20 / self.sigma))
        eval_points = (eval_points // 2) + 1
        values = smooth_round_to_grid(f, y, eval_points = eval_points)
        smoothed = self.convolve(values, eval_points)
        return interpolate(x_eval, smoothed)

    def kde(self, x, x_eval):
        weights = np.ones_like(x) / len(x)
        norm = sp.stats.norm(scale=self.sigma)
        Int = norm.sf(-x) - norm.sf(1-x) # Pr[ \eta_\sigma \in [-x, 1-x] ]
        weights /= Int
        density = self.apply(x, weights, x_eval)
        return density


def smooth_round_to_grid(f, y, eval_points = 1000):
    values = np.zeros(eval_points)
    bins = (f * (eval_points-1)).astype(int).clip(0, eval_points-2)
    frac = f * (eval_points - 1) - bins
    np.add.at(values, bins, (1-frac)*y)
    np.add.at(values, bins + 1, frac*y)
    return values
