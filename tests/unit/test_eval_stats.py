"""Paired statistics: McNemar, Holm, bootstrap (hand-computed expectations)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from copilot.evaluation.stats import (
    bootstrap_ci,
    bootstrap_indices,
    exact_mcnemar,
    holm_adjust,
    paired_table,
)


def test_paired_table_counts_each_cell_and_the_signed_difference():
    x = [True, True, False, False, True]
    y = [True, False, True, False, False]
    t = paired_table(x, y)
    assert (t.both, t.only_x, t.only_y, t.neither) == (1, 2, 1, 1)
    assert t.n == 5 and t.discordant == 3
    assert t.difference == pytest.approx((2 - 1) / 5)
    with pytest.raises(ValueError):
        paired_table([True], [True, False])


def test_one_question_is_2_94_points_of_34():
    t = paired_table([True] + [False] * 33, [False] * 34)
    assert 100 * t.difference == pytest.approx(2.941, abs=1e-3)


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        (0, 5, 2 / 32),  # n=5, tail = C(5,0)
        (5, 0, 2 / 32),  # symmetric
        (3, 7, 2 * (1 + 10 + 45 + 120) / 1024),  # n=10
        (5, 5, 1.0),  # balanced -> capped at 1
        (0, 0, 1.0),  # no discordant pairs
        (1, 0, 1.0),  # n=1: 2*1/2 = 1
        (0, 10, 2 / 1024),
    ],
)
def test_exact_mcnemar_matches_hand_computed_binomial_tails(b, c, expected):
    assert exact_mcnemar(b, c) == pytest.approx(expected)


def test_exact_mcnemar_equals_the_binomial_definition_for_many_counts():
    for b in range(0, 12):
        for c in range(0, 12):
            n = b + c
            if n == 0:
                continue
            k = min(b, c)
            tail = sum(math.comb(n, i) * 0.5**n for i in range(k + 1))
            assert exact_mcnemar(b, c) == pytest.approx(min(1.0, 2 * tail))
    with pytest.raises(ValueError):
        exact_mcnemar(-1, 2)


def test_holm_matches_a_hand_worked_example_and_keeps_order_of_keys():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})
    assert list(adjusted) == ["a", "b", "c"]
    assert holm_adjust({"only": 0.2}) == {"only": 0.2}
    assert holm_adjust({"x": 0.9, "y": 0.8}) == {"x": 1.0, "y": 1.0}  # capped at 1


def test_holm_never_lowers_a_p_value_and_stays_in_unit_interval():
    p = {f"t{i}": v for i, v in enumerate([0.001, 0.2, 0.049, 0.5, 0.03])}
    adjusted = holm_adjust(p)
    assert all(p[k] <= adjusted[k] <= 1.0 for k in p)


def test_bootstrap_indices_are_deterministic_for_a_seed_and_differ_between_seeds():
    a = bootstrap_indices(34, 1000, seed=7)
    assert a.shape == (1000, 34) and a.min() >= 0 and a.max() < 34
    assert np.array_equal(a, bootstrap_indices(34, 1000, seed=7))
    assert not np.array_equal(a, bootstrap_indices(34, 1000, seed=8))
    with pytest.raises(ValueError):
        bootstrap_indices(0, 10, 1)


def test_bootstrap_ci_is_degenerate_for_constant_data_and_brackets_the_mean():
    idx = bootstrap_indices(20, 4000, seed=1)
    assert bootstrap_ci([0.5] * 20, idx) == (0.5, 0.5)
    values = [1.0] * 12 + [0.0] * 8
    lo, hi = bootstrap_ci(values, idx)
    assert lo < 0.6 < hi and lo >= 0.0 and hi <= 1.0
    narrower = bootstrap_ci(values, idx, confidence=0.5)
    assert lo <= narrower[0] and narrower[1] <= hi
    with pytest.raises(ValueError):
        bootstrap_ci([1.0] * 5, idx)  # wrong number of questions
    with pytest.raises(ValueError):
        bootstrap_ci(values, idx, confidence=1.0)


def test_the_same_resample_is_shared_by_every_metric():
    idx = bootstrap_indices(10, 500, seed=3)
    x = list(range(10))
    lo1, hi1 = bootstrap_ci(x, idx)
    lo2, hi2 = bootstrap_ci([v * 2 for v in x], idx)
    assert (lo2, hi2) == pytest.approx((2 * lo1, 2 * hi1))  # perfectly correlated resampling
