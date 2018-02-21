# -*- coding: utf-8 -*-
"""Metrics."""
from __future__ import division

import logging

import numpy as np
from scipy.sparse import issparse
from sklearn.utils.multiclass import type_of_target


def precision_k_score(y_true, y_pred, k=10):
    """Calculate precision@k for specified value of k."""
    return precision_k_curve(y_true, y_pred, max_k=k)[-1]


def precision_k_curve(y_true, y_pred, max_k=0):
    """
    Calculate precision@k for various values of k.

    y_true indicates the products that the user actually interacted with. it must be a 'multilabel-indicator'.

    If max_k is zero, precision@k is calculated for all values of k from 1 to number of items. Otherwise, it is
    calculated for all values of k from 1 to max_k.
    """
    y_true, y_pred = _check_targets(y_true, y_pred)
    _, item_count = y_true.shape
    if not max_k:
        max_k = item_count

    if max_k < 1 or max_k > item_count:
        raise ValueError('max_k should be at least 1 and less than or equal to number of labels')

    logging.debug("calculating precision@k for k=1 to k=%s", max_k)

    return _precision_curve(y_true, y_pred, max_k)


def _precision_curve(y_true, y_pred, max_k):
    """Return tuple consisting of precision curve using sample average and micro average respectively."""
    # Find the top items among the predicted
    top_pred_indices = np.argsort(y_pred)[:, ::-1][:, :max_k]
    curve = np.repeat(0.0, max_k)

    user_count, _ = y_true.shape
    for user_index in range(user_count):
        y_true_user = y_true[user_index, :]
        if issparse(y_true_user):
            y_true_user = y_true_user.todense().A1
        user_positives = set(np.where(y_true_user > 0.0)[0])
        hits = np.repeat(0.0, max_k)
        user_top_pred_indices = top_pred_indices[user_index, :]

        for position, current_pred_index in enumerate(user_top_pred_indices):
            # found_at_positions = np.where(user_top_pred_indices == current_rated_index)[0]
            if current_pred_index not in user_positives:
                continue
            hits[position] += 1
        curve += hits

    curve = np.cumsum(curve) / (np.arange(1, max_k + 1, dtype=float) * user_count)
    return curve


def recall_k_score(y_true, y_pred, average='samples', k=10):
    """Calculate recall@k for specified value of k."""
    return recall_k_curve(y_true, y_pred, average, max_k=k)[-1]


def recall_k_curve(y_true, y_pred, average='samples', max_k=0):
    """
    Calculate recall@k for various values of k.

    y_true indicates the products that the user actually interacted with. it must be a 'multilabel-indicator'.
    Parameter average can take values 'samples' or 'micro'.

    If max_k is zero, recall@k is calculated for all values of k from 1 to number of items. Otherwise, it is
    calculated for all values of k from 1 to max_k.
    """
    average_options = ('micro', 'samples')
    if average not in average_options:
        raise ValueError('average has to be one of ' + str(average_options))

    y_true, y_pred = _check_targets(y_true, y_pred)
    _, item_count = y_true.shape
    if not max_k:
        max_k = item_count

    if max_k < 1 or max_k > item_count:
        raise ValueError('max_k should be at least 1 and less than or equal to number of labels')

    logging.debug("calculating recall@k for k=1 to k=%s, average: %s", max_k, average)

    sample_curve, micro_curve = _recall_curves(y_true, y_pred, max_k)

    return sample_curve if average == 'samples' else micro_curve


def _recall_curves(y_true, y_pred, max_k):
    """Return tuple consisting of recall curve using sample average and micro average respectively."""
    # Find the top items among the predicted
    top_pred_indices = np.argsort(y_pred)[:, ::-1][:, :max_k]
    sample_curve = np.repeat(0.0, max_k)
    micro_curve = np.repeat(0.0, max_k)
    total_ratings = 0

    user_count, _ = y_true.shape
    for user_index in range(user_count):
        y_true_user = y_true[user_index, :]
        if issparse(y_true_user):
            y_true_user = y_true_user.todense().A1
        rated_indices = np.where(y_true_user > 0.0)[0]
        hits = np.repeat(0.0, max_k)
        user_top_pred_indices = top_pred_indices[user_index, :]

        for current_rated_index in rated_indices:
            found_at_positions = np.where(user_top_pred_indices == current_rated_index)[0]
            if not found_at_positions.shape[0]:
                continue
            hit_at_position = found_at_positions[0]
            hits[hit_at_position] += 1
        micro_curve += hits
        total_ratings += len(rated_indices)
        sample_curve += (hits / len(rated_indices))

    sample_curve = sample_curve / user_count
    sample_curve = np.cumsum(sample_curve)
    micro_curve = micro_curve / total_ratings
    micro_curve = np.cumsum(micro_curve)

    return sample_curve, micro_curve


def bag_recall_k_score(y_true, y_pred, average='samples', bag_size=1000, k=10, seed=None):
    """Calculate bag recall@k for specified value of k."""
    return bag_recall_k_curve(y_true, y_pred, average, bag_size, seed)[k - 1]


def bag_recall_k_curve(y_true, y_pred, average='samples', bag_size=1000, seed=None):
    """Calculate bag recall@k for all ks up to bag size + 1."""
    average_options = ('micro', 'samples')
    if average not in average_options:
        raise ValueError('average has to be one of ' + str(average_options))

    y_true, y_pred = _check_targets(y_true, y_pred)
    user_count, item_count = y_true.shape

    if bag_size < 1 or bag_size > item_count:
        raise ValueError('bag_size should be at least 1 and less than or equal to number of labels')

    logging.debug("calculating bag-recall@k for k=1 to k=%s, average: %s", bag_size, average)

    bag_recall_curve = np.repeat(0.0, bag_size + 1)
    bag_distribution_curve = np.repeat(0.0, bag_size + 1)
    total_ratings = 0
    np.random.seed(seed)

    user_count, _ = y_true.shape
    for user_index in range(user_count):
        hits, rated_count = _bag_recall_for_user(user_index, y_true, y_pred, bag_size)
        bag_recall_curve += (hits / rated_count)
        bag_distribution_curve += hits
        total_ratings += rated_count

    bag_recall_curve = bag_recall_curve / user_count
    bag_recall_curve = np.cumsum(bag_recall_curve)
    bag_distribution_curve = bag_distribution_curve / total_ratings
    bag_distribution_curve = np.cumsum(bag_distribution_curve)

    return bag_recall_curve if average == 'samples' else bag_distribution_curve


def _bag_recall_for_user(user_index, y_true, y_pred, bag_size):
    y_true_user = y_true[user_index, :]
    if issparse(y_true_user):
        y_true_user = y_true_user.todense().A1

    unrated_indices = np.where(y_true_user == 0.0)[0]
    rated_indices = np.where(y_true_user > 0.0)[0]
    hits = np.repeat(0.0, bag_size + 1)

    for current_rated_index in rated_indices:
        bag = np.random.choice(unrated_indices, bag_size, replace=False)
        bag = np.append(bag, current_rated_index)
        bag_predictions = y_pred[user_index, bag]
        top_predicted_indices = bag[np.argsort(bag_predictions)[::-1]]
        temp = np.where(top_predicted_indices == current_rated_index)[0]

        if not temp.shape[0]:
            continue
        hit_at_position = temp[0]
        hits[hit_at_position] += 1
    return hits, len(rated_indices)


def _check_targets(y_true, y_pred):
    if not issparse(y_true):
        y_true = np.asarray(y_true)

    if issparse(y_pred):
        y_pred = y_pred.toarray()
    else:
        y_pred = np.asarray(y_pred)

    if type_of_target(y_true) != 'multilabel-indicator':
        raise ValueError('y_true should be of type multilabel-indicator')

    if type_of_target(y_pred) != 'continuous-multioutput':
        raise ValueError('y_pred should be of type continuous-multioutput')

    return y_true, y_pred


def mean_rank_score(y_true, y_pred, average='micro'):
    """Calculate mean percentile rank. Lower is better. See http://yifanhu.net/PUB/cf.pdf for details."""
    average_options = ('micro', 'samples')
    if average not in average_options:
        raise ValueError('average has to be one of ' + str(average_options))

    y_true, y_pred = _check_targets(y_true, y_pred)

    logging.debug("calculating mean rank score, average: %s", average)

    user_count, item_count = y_true.shape

    # Sort the items descending by their score
    sorted_pred_indices = np.argsort(y_pred)[:, ::-1]

    sample_score = 0.0
    micro_score = 0.0
    total_ratings = 0

    for user_index in range(user_count):
        y_true_user = y_true[user_index, :]
        if issparse(y_true_user):
            y_true_user = y_true_user.todense().A1
        rated_indices = np.where(y_true_user > 0.0)[0]
        user_sorted_pred_indices = sorted_pred_indices[user_index, :]

        user_score = 0.0
        for current_rated_index in rated_indices:
            found_at_positions = np.where(user_sorted_pred_indices == current_rated_index)[0]
            assert found_at_positions.shape[0]
            hit_at_position = found_at_positions[0]
            user_score += hit_at_position / item_count

        sample_score += user_score / len(rated_indices)
        micro_score += user_score
        total_ratings += len(rated_indices)

    sample_score = sample_score / user_count
    micro_score = micro_score / total_ratings

    return sample_score if average == 'samples' else micro_score


def ndcg_k_curve(y_true, y_pred, max_k=0):
    y_true, y_pred = _check_targets(y_true, y_pred)

    if issparse(y_true):
        y_true = y_true.toarray()

    user_count, item_count = y_true.shape
    if not max_k:
        max_k = item_count

    logging.debug("calculating ndcg@k for k=1 to k=%s", max_k)

    sorted_pred_indices = np.argsort(y_pred)[:, ::-1][:, :max_k]
    sorted_ideal_indices = np.argsort(y_true)[:, ::-1][:, :max_k]

    running_ndcg = 0.0
    for user_index in range(user_count):
        y_true_user = y_true[user_index, :]
        relevance_predicted_rank = y_true_user[sorted_pred_indices[user_index, :]]
        dcg_actual = _dcg(relevance_predicted_rank, max_k)

        relevance_ideal_rank = y_true_user[sorted_ideal_indices[user_index, :]]
        dcg_ideal = _dcg(relevance_ideal_rank, max_k)

        running_ndcg += (dcg_actual / dcg_ideal)

    return running_ndcg / user_count


def _dcg(relevance, k):
    gains = np.power(2, relevance[:k]) - 1
    discounts = np.log2(np.arange(k) + 2)
    return np.cumsum(gains / discounts)
