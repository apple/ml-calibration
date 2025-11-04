#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import numpy as np
import scipy as sp

from . import config
from .kernels import LogitGaussianKernel, smooth_round_to_grid, ReflectedGaussianKernel

def _get_default_kernel():
    if config.use_logit_scaling:
        return LogitGaussianKernel
    else:
        return ReflectedGaussianKernel

def binning(f, y, bin_size=0.1, shift=0):
    bi = (f + shift) / bin_size
    bi = bi.astype(int)
    r = f - y
    bins_cnt = int((shift + 1) / bin_size) + 1
    bins = np.zeros(bins_cnt)
    np.add.at(bins, bi, r)
    return bins


def intCE_rand(f, y, eps=0.001, trials=100):
    bin_size = 1.0
    min_error = 3
    while bin_size > eps:
        binning_error = 0.0
        for _ in range(trials):  # trials to compute expectation of RintCE
            shift = np.random.uniform(0, bin_size)
            B = binning(f, y, bin_size=bin_size, shift=shift)
            binning_error += abs(B).sum() / len(f)
        binning_error /= trials

        min_error = min(min_error, binning_error + bin_size)
        bin_size = bin_size / (2)
    return min_error

def smooth_ece(f, y, bin_size):
    ev_points = config.smECE_mesh_pts
    kernel = _get_default_kernel()
    ev_points = max(round(10 / bin_size), ev_points)
    t = np.linspace(0, 1, ev_points)
    ker = kernel(bin_size)
    rs, density = ker.smooth(f, (f - y), t)
    return np.sum(np.abs(rs) * density) / np.sum(density)

def search_param(predicate, start=1, refine=10):
    if predicate(start):
        return start
    start, end = 1, 0 
    for _ in range(refine):
        midpoint = (start + end) / 2
        if predicate(midpoint):
            end = midpoint
        else:
            start = midpoint
    return start

def smECE_sigma(f, y, sigma=0.05):
    """
        Computes SmoothECE at a fixed bandwidth sigma.
    """
    return smooth_ece(f, y, sigma)

def smECE(f, y, eps=0.001, return_width=False):
    """
        Computes SmoothECE with automatic choice of bandwidth sigma, as described in [Błasiok-Nakkiran '23].
        Note, the implementation discretizes the kernel for large datasets.
    """
    def check_smooth_ece(alpha):
        return alpha < eps or alpha < smooth_ece(f, y, alpha)

    bin_size = search_param(check_smooth_ece, start=1, refine=10)
    if return_width:
        return smooth_ece(f, y, bin_size), bin_size
    else:
        return smooth_ece(f, y, bin_size)

smECE_fast = smECE # for backwards compatability

def binnedECE(f, y, nbins=10):
    bins = binning(f, y, bin_size=1.0 / nbins, shift=0)
    return np.abs(bins).sum() / len(f)


def binnedECEw(f, y, nbins=10):
    return binnedECE(f, y, nbins) + 1.0 / nbins


def laplace_calibration_approx(f, y, terms=None, theta=1.0):
    if terms is None:
        terms = 10 * len(f)
    terms = max(terms, 5000)  # at least 5000 terms
    residual = f - y

    left = np.random.randint(len(f) - 1, size=terms)
    right = np.random.randint(len(f) - 1, size=terms)

    r_1 = np.take_along_axis(residual, left, 0)
    r_2 = np.take_along_axis(residual, right, 0)
    f_1 = np.take_along_axis(f, left, 0)
    f_2 = np.take_along_axis(f, right, 0)
    ker = np.exp(-np.abs(f_1 - f_2) / theta)
    return np.sqrt(np.abs((1 / theta) * (r_1 * r_2 * ker).sum() / terms))

def multiclass_logits_to_confidences(f_logits, y_true, probs=False):
    """
        f_logits: [N, C] array of predicted logits over C-classes.
        y_true:   [N, 1] array of true labels
        probs (optional): Whether the input is probabilities (default assumes logits).
        
        returns: (f, y) binarized (confidence, accuracy) pairs corresponding to confidence-calibration
    """
    f_logits, y_true = np.array(f_logits), np.array(y_true)
    if not probs:
        f_probs = sp.special.softmax(f_logits, axis=-1)
    else:
        f_probs = f_logits
    f_pred = f_probs.argmax(axis=-1)
    correct = (f_pred == y_true)*1.0
    max_prob = f_probs.max(axis=-1).reshape(-1)
    return max_prob, correct
