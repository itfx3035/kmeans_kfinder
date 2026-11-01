'''Behaviour tests for KMeansKFinder.

Run against any implementation:

    KFINDER_MODULE=kmeans_kfinder pytest tests/                       # current code
    KFINDER_MODULE=baseline.kmeans_kfinder_baseline pytest tests/     # pre-change code

The tests describe the documented behaviour of the project, so the tests marked
EXPECTED BEHAVIOUR fail on the pre-change code on purpose: those failures are the
measurement of the improvement. Tests marked REGRESSION GUARD must pass on both.
'''

import importlib
import json
import os
import time

import numpy as np
import pytest

from sample_dataset import get_dataset

MODULE_NAME = os.environ.get('KFINDER_MODULE', 'kmeans_kfinder')
KFinder = importlib.import_module(MODULE_NAME).KMeansKFinder

EXPECTED = json.load(open(os.path.join(os.path.dirname(__file__), 'expected_baseline.json')))


def make(X, max_k=30, **kwargs):
    return KFinder(X, max_k=max_k, random_state=1, **kwargs)


def documented_vote(opts):
    '''Majority vote, largest k wins ties (documented rule).'''
    values = [v for v in opts.values() if v is not None]
    if not values:
        return None
    counts = {v: values.count(v) for v in values}
    top = max(counts.values())
    return max(v for v, c in counts.items() if c == top)


# --------------------------------------------------- regression guards (must pass on both)

def test_clean_data_finds_ground_truth():
    X, ground_truth_k = get_dataset('clean_blobs')
    assert make(X).find_best_k() == ground_truth_k


def test_results_match_pre_change_code_on_valid_data():
    '''Improvements must not change answers that the old code got right.'''
    for name, expected in EXPECTED.items():
        if expected['best_k'] is None:
            continue
        X, _ = get_dataset(name)
        assert make(X).find_best_k() == expected['best_k'], name


def test_iris_is_reproducible():
    '''Iris has overlapping classes, so elbow methods may disagree with k=3, but the
    answer must be reproducible for a fixed random_state.'''
    X, _ = get_dataset('iris')
    finder = make(X)
    best_k = finder.find_best_k()
    assert 1 <= best_k <= 30 and best_k == finder.find_best_k() == make(X).find_best_k()


# ---------------------------------------------------- expected behaviour (fails before)

def test_fit_best_without_calling_find_best_k():
    '''fit_best() promises to compute best_k if it was not calculated yet.'''
    X, ground_truth_k = get_dataset('clean_blobs')
    km = make(X).fit_best()
    assert int(km.n_clusters) == ground_truth_k


def test_voting_follows_documented_rule():
    '''Majority vote must break ties by taking the larger k.'''
    # single_cluster_noise with seed 3 produces a two-method tie (21 vs 7)
    X, _ = get_dataset('single_cluster_noise')
    finder = KFinder(X, max_k=30, random_state=3)
    best_k = finder.find_best_k()
    assert best_k == documented_vote(finder.best_k_opts), (
        f'voting rule violated: opts={finder.best_k_opts}, got {best_k}')


def test_all_five_methods_report_on_clean_data():
    X, ground_truth_k = get_dataset('clean_blobs')
    finder = make(X)
    finder.find_best_k()
    assert len(finder.best_k_opts) == 5
    assert documented_vote(finder.best_k_opts) == ground_truth_k


# ---------------------------------------------------- robustness (fails before)

def test_zero_variance_data_does_not_raise():
    finder = make(np.ones((100, 2)))
    finder.find_best_k()          # must not raise ZeroDivisionError


def test_degenerate_identical_points_do_not_raise():
    X, _ = get_dataset('degenerate_identical')
    finder = make(X)
    finder.find_best_k()          # must not raise ZeroDivisionError


def test_max_k_larger_than_n_samples_does_not_raise():
    X, _ = get_dataset('tiny_dataset')
    best_k = make(X, max_k=30).find_best_k()
    assert best_k is None or 1 <= best_k <= X.shape[0]


def test_max_k_one_is_handled():
    X, _ = get_dataset('clean_blobs')
    best_k = make(X, max_k=1).find_best_k()
    assert best_k is None or best_k >= 1


def test_invalid_max_k_is_rejected():
    X, _ = get_dataset('clean_blobs')
    for bad in (0, -3, 2.5):
        try:
            KFinder(X, max_k=bad)
        except ValueError:
            continue
        pytest.fail(f'max_k={bad} should raise ValueError')


# ---------------------------------------------------- side effects (fail before)

def test_caller_kwargs_are_not_mutated():
    X, _ = get_dataset('clean_blobs')
    caller_kwargs = {'random_state': 1, 'n_init': 3}
    finder = KFinder(X, max_k=5, **caller_kwargs)
    finder.find_best_k()
    assert caller_kwargs == {'random_state': 1, 'n_init': 3}, caller_kwargs


def test_find_best_k_is_repeatable():
    X, ground_truth_k = get_dataset('clean_blobs')
    finder = make(X)
    assert finder.find_best_k() == finder.find_best_k() == ground_truth_k


def test_best_model_is_cached():
    X, ground_truth_k = get_dataset('clean_blobs')
    finder = make(X, max_k=10)
    finder.fit_best()
    assert finder.fit_best() is finder.fit_best()
    assert int(finder.fit_best().n_clusters) == ground_truth_k


def test_stable_across_random_seeds_on_clean_data():
    X, ground_truth_k = get_dataset('clean_blobs')
    results = [KFinder(X, max_k=30, random_state=s).find_best_k() for s in range(1, 6)]
    assert results.count(ground_truth_k) >= 4, results


# ---------------------------------------------------- measurement

def test_measurement_report(capsys):
    X, ground_truth_k = get_dataset('clean_blobs')
    t0 = time.perf_counter()
    finder = make(X, max_k=30)
    best_k = finder.find_best_k()
    elapsed = (time.perf_counter() - t0) * 1000
    seeds = [KFinder(X, max_k=30, random_state=s).find_best_k() for s in range(1, 6)]
    print(f'[{MODULE_NAME}] clean_blobs truth={ground_truth_k} best_k={best_k} '
          f'opts={finder.best_k_opts} seeds={seeds} time={elapsed:.1f}ms')
    assert len(seeds) == 5
