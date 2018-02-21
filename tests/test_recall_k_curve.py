# -*- coding: utf-8 -*-
"""Tests for recall@k curve."""
# pylint: disable=protected-access,too-many-public-methods,no-self-use,too-few-public-methods,C0103

from __future__ import division

import numpy as np
import pandas as pd

from rankmetrics.metrics import recall_k_curve


class TestRecallKCurve(object):
    """Test class for the recall_k_curve method."""

    def test_recall_k_curve_list(self):
        """Test recall_k_curve on a list."""
        y_true = [[0, 0, 0, 1, 1]]
        y_pred = [[0.1, 0.0, 0.0, 0.2, 0.3]]
        actual = recall_k_curve(y_true, y_pred, max_k=3)
        expected = [0.5, 1., 1.]
        np.testing.assert_allclose(expected, actual)

    def test_recall_k_curve_np(self):
        """Test recall_k_curve on a numpy array."""
        y_true = np.array([[0, 0, 0, 1, 1]])
        y_pred = np.array([[0.1, 0.0, 0.0, 0.2, 0.3]])
        actual = recall_k_curve(y_true, y_pred, max_k=3)
        expected = [0.5, 1., 1.]
        np.testing.assert_allclose(expected, actual)

    def test_recall_k_curve_pd(self):
        """Test recall_k_curve on a pandas data fame."""
        y_true = pd.DataFrame([[0, 0, 0, 1, 1]])
        y_pred = pd.DataFrame([[0.1, 0.0, 0.0, 0.2, 0.3]])
        actual = recall_k_curve(y_true, y_pred, max_k=3)
        expected = [0.5, 1., 1.]
        np.testing.assert_allclose(expected, actual)

    def test_recall_k_curve_multiple_list(self):
        """Test recall_k_curve on a list."""
        y_true = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]
        y_pred = [[0.1, 0.0, 0.0, 0.2, 0.3], [0.1, 0.0, 0.0, 0.2, 0.3]]
        actual = recall_k_curve(y_true, y_pred, max_k=3)
        expected = [5/12, 5/6, 5/6]
        np.testing.assert_allclose(expected, actual)

    def test_recall_k_curve_multiple_list_micro(self):
        """Test recall_k_curve on a list."""
        y_true = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]
        y_pred = [[0.1, 0.0, 0.0, 0.2, 0.3], [0.1, 0.0, 0.0, 0.2, 0.3]]
        actual = recall_k_curve(y_true, y_pred, max_k=3, average='micro')
        expected = [2/5, 4/5, 4/5]
        np.testing.assert_allclose(expected, actual)
