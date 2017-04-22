# -*- coding: utf-8 -*-
"""Tests for recall@k score."""
# pylint: disable=protected-access,too-many-public-methods,no-self-use,too-few-public-methods,C0103

import numpy as np
from scipy.sparse import csr_matrix
from rankmetrics.metrics import recall_k_score


class TestRecallKScore(object):
    """Test class for the recall_k_score method."""

    def test_recall_k_score_list_k1(self):
        """Test recall_k_score on a list."""
        y_true = [[0, 0, 0, 1, 1]]
        y_pred = [[0.1, 0.0, 0.0, 0.2, 0.3]]
        actual = recall_k_score(y_true, y_pred, k=1)
        expected = 0.5
        np.testing.assert_allclose(expected, actual)

    def test_recall_k_score_list_k2(self):
        """Test recall_k_score on a list."""
        y_true = [[0, 0, 0, 1, 1]]
        y_pred = [[0.1, 0.0, 0.0, 0.2, 0.3]]
        actual = recall_k_score(y_true, y_pred, k=2)
        expected = 1.0
        np.testing.assert_allclose(expected, actual)

    def test_recall_k_score_micro_k1_csr_np(self):
        """Test bag_recall_k_score on a list."""
        y_true = csr_matrix([[0, 0, 0, 10, 12], [0, 0, 2, 13, 30]])
        y_true = y_true > 0
        y_pred = np.array([[0.1, 0.0, 0.0, 0.2, 0.3], [0.1, 0.0, 0.0, 0.2, 0.3]])
        print("y_true", type(y_true), y_true.shape)
        print("y_pred", type(y_pred), y_pred.shape)

        actual = recall_k_score(y_true, y_pred, k=1)
        print(actual)
        expected = 0.416666666667
        np.testing.assert_allclose(expected, actual)
