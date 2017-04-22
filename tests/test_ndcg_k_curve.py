# -*- coding: utf-8 -*-
"""Tests for NCDG@k curve."""
# pylint: disable=protected-access,too-many-public-methods,no-self-use,too-few-public-methods,C0103

import numpy as np
import pandas as pd
from rankmetrics.metrics import ndcg_k_curve


class TestRecallKCurve(object):
    """Test class for the recall_k_curve method."""

    def test_ndcg_k_curve_pd(self):
        """Test ndcg_k_curve on a pandas data fame."""
        y_true = pd.DataFrame([[1, 0, 0, 1, 0, 0]])
        y_pred = pd.DataFrame([[0.1, 0.0, 0.0, 0.2, 0.3, 0.0]])

        actual = ndcg_k_curve(y_true, y_pred, max_k=6)
        expected = [0, 0.386852807234542, 0.693426403617271, 0.693426403617271, 0.693426403617271, 0.693426403617271]
        np.testing.assert_allclose(expected, actual)
