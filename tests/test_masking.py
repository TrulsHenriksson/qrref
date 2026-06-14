import pytest

import numpy as np

from qrpy.masking import (
    pattern_masks,
    run_lengths,
    run_length_penalty,
    block_penalty,
    bad_pattern_penalty,
    proportion_penalty,
    N_1,
    N_2,
    N_3,
    N_4,
)

@pytest.mark.parametrize("values, expected", [
    ([0, 1, 1, 1, 0, 0, 1, 1, 1, 1], [1, 3, 2, 4]),
    ([0, 0], [2]),
    ([0], [1]),
    ([], [0]),
])
def test_run_lengths(values, expected):
    assert np.array_equal(run_lengths(np.array(values)), np.array(expected))


@pytest.fixture
def sample_symbol():
    return np.array([
        [1, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
    ], dtype=np.bool)

def test_run_length_penalty(sample_symbol):
    runs_over_5 = np.array([5, 5, 6])
    assert run_length_penalty(sample_symbol) == np.sum(runs_over_5 - 5 + N_1)

def test_block_penalty(sample_symbol):
    blocks = 8
    assert block_penalty(sample_symbol) == blocks * N_2

def test_bad_pattern_penalty():
    bad_symbol = np.array([
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.bool)
    for k in range(4):
        symbol = np.rot90(bad_symbol, k)
        assert bad_pattern_penalty(symbol) == N_3
        assert bad_pattern_penalty(symbol.T) == N_3

def test_proportion_penalty():
    arange = np.arange(100).reshape((10, 10))
    for n in range(1, 100):
        diff_from_50 = abs(n - 50)
        assert proportion_penalty(arange < n) == N_4 * (diff_from_50 // 5), f"{n = } failed"
