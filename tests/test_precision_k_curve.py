# -*- coding: utf-8 -*-
"""Tests for precision@k curve."""
# pylint: disable=protected-access,too-many-public-methods,no-self-use,too-few-public-methods,C0103

import numpy as np
import pandas as pd
from rankmetrics.metrics import precision_k_curve


class TestPrecisionKCurve(object):
    """Test class for the precision_k_curve method."""

    def test_precision_k_curve_list(self):
        """Test precision_k_curve on a list."""
        y_true = [[0, 0, 0, 1, 1]]
        y_pred = [[0.1, 0.0, 0.0, 0.2, 0.3]]
        actual = precision_k_curve(y_true, y_pred, max_k=3)
        expected = [1, 1., 2/3]
        np.testing.assert_allclose(expected, actual)

    def test_precision_k_curve_np(self):
        """Test precision_k_curve on a numpy array."""
        y_true = np.array([[0, 0, 0, 1, 1]])
        y_pred = np.array([[0.1, 0.0, 0.0, 0.2, 0.3]])
        actual = precision_k_curve(y_true, y_pred, max_k=3)
        expected = [1, 1., 2/3]
        np.testing.assert_allclose(expected, actual)

    def test_precision_k_curve_pd(self):
        """Test precision_k_curve on a pandas data fame."""
        y_true = pd.DataFrame([[0, 0, 0, 1, 1]])
        y_pred = pd.DataFrame([[0.1, 0.0, 0.0, 0.2, 0.3]])
        actual = precision_k_curve(y_true, y_pred, max_k=3)
        expected = [1, 1., 2/3]
        np.testing.assert_allclose(expected, actual)

    def test_precision_k_curve_multiple_list(self):
        """Test precision_k_curve on a list."""
        y_true = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]
        y_pred = [[0.1, 0.0, 0.0, 0.2, 0.3], [0.1, 0.0, 0.0, 0.2, 0.3]]
        actual = precision_k_curve(y_true, y_pred, max_k=3)
        expected = [2/2, 4/4, 4/6]
        np.testing.assert_allclose(expected, actual)
