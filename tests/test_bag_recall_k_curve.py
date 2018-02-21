# -*- coding: utf-8 -*-
"""Tests for recall@k curve."""
# pylint: disable=protected-access,too-many-public-methods,no-self-use,too-few-public-methods,C0103

from __future__ import division

import numpy as np

from rankmetrics.metrics import bag_recall_k_curve


class TestBagRecallKCurve(object):
    """Test class for the bag_recall_k_curve method."""

    def test_bag_recall_k_curve_list(self):
        """Test recall_k_curve on a list."""
        y_true = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]
        y_pred = [[0.1, 0.0, 0.0, 0.2, 0.3], [0.1, 0.0, 0.0, 0.2, 0.3]]
        actual = bag_recall_k_curve(y_true, y_pred, bag_size=2, seed=0)
        expected = [5/6, 1., 1.]
        np.testing.assert_allclose(expected, actual)

    def test_bag_recall_k_curve_list_micro(self):
        """Test recall_k_curve on a list."""
        y_true = [[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]]
        y_pred = [[0.1, 0.0, 0.0, 0.2, 0.3], [0.1, 0.0, 0.0, 0.2, 0.3]]
        actual = bag_recall_k_curve(y_true, y_pred, bag_size=2, average='micro', seed=0)
        expected = [4/5, 1., 1.]
        np.testing.assert_allclose(expected, actual)
