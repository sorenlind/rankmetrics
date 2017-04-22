# -*- coding: utf-8 -*-
"""Tests for bag recall@k score."""
# pylint: disable=protected-access,too-many-public-methods,no-self-use,too-few-public-methods,C0103

import numpy as np
from scipy.sparse import csr_matrix
from rankmetrics.metrics import bag_recall_k_score


class TestBagRecallKScore(object):
    """Test class for the bag_recall_k_score method."""

    def test_bag_recall_k_score_samples_k1(self):
        """Test bag_recall_k_score on a list."""
        y_true = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]
        y_pred = [[0.1, 0.0, 0.0, 0.2, 0.3], [0.1, 0.0, 0.0, 0.2, 0.3]]
        actual = bag_recall_k_score(y_true, y_pred, bag_size=2, seed=0, k=1)
        expected = 5/6
        np.testing.assert_allclose(expected, actual)

    def test_bag_recall_k_score_samples_k2(self):
        """Test bag_recall_k_score on a list."""
        y_true = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]
        y_pred = [[0.1, 0.0, 0.0, 0.2, 0.3], [0.1, 0.0, 0.0, 0.2, 0.3]]
        actual = bag_recall_k_score(y_true, y_pred, bag_size=2, seed=0, k=2)
        expected = 1.
        np.testing.assert_allclose(expected, actual)

    def test_bag_recall_k_score_micro_k1(self):
        """Test bag_recall_k_score on a list."""
        y_true = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]
        y_pred = [[0.1, 0.0, 0.0, 0.2, 0.3], [0.1, 0.0, 0.0, 0.2, 0.3]]
        actual = bag_recall_k_score(y_true, y_pred, bag_size=2, average='micro', seed=0, k=1)
        expected = 4/5
        np.testing.assert_allclose(expected, actual)

    def test_bag_recall_k_score_micro_k2(self):
        """Test bag_recall_k_score on a list."""
        y_true = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]
        y_pred = [[0.1, 0.0, 0.0, 0.2, 0.3], [0.1, 0.0, 0.0, 0.2, 0.3]]
        actual = bag_recall_k_score(y_true, y_pred, bag_size=2, average='micro', seed=0, k=2)
        expected = 1.
        np.testing.assert_allclose(expected, actual)

    def test_bag_recall_k_score_micro_k1_csr_np(self):
        """Test bag_recall_k_score on a list."""
        y_true = csr_matrix([[0, 0, 0, 10, 12], [0, 0, 2, 13, 30]])
        y_true = y_true > 0
        y_pred = np.array([[0.1, 0.0, 0.0, 0.2, 0.3], [0.1, 0.0, 0.0, 0.2, 0.3]])
        actual = bag_recall_k_score(y_true, y_pred, bag_size=2, average='micro', seed=0, k=1)
        expected = 4/5
        np.testing.assert_allclose(expected, actual)
