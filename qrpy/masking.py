import numpy as np

from qrpy.custom_types import Symbol
from qrpy.table_data import modules_per_side
from qrpy.placement import get_data_mask


# Base penalty for a run of same-colored modules of length >= 5
N_1 = 3
# Penalty for 2x2 blocks of the same color
N_2 = 3
# Penalty for existence of 1:1:3:1:1 pattern of dark-light-dark-light-dark modules
N_3 = 40
# Penalty for unbalanced light-dark ratio
N_4 = 10


def pattern_masks(version: int) -> list[Symbol]:
    """Generate the patterns used for masking.

    Implements Table 10, page 50.
    """
    width = modules_per_side(version)
    # uint16 so that axis_indices[-1]**2 doesn't overflow even for version=40.
    axis_indices = np.arange(width, dtype=np.uint16)
    ROW, COL = np.meshgrid(axis_indices, axis_indices, indexing="ij")
    SUM = ROW + COL
    PROD = ROW * COL
    return [
        SUM % 2 == 0,
        ROW % 2 == 0,
        COL % 3 == 0,
        SUM % 3 == 0,
        (ROW // 2 + COL // 3) % 2 == 0,
        (a := PROD % 2 + PROD % 3) == 0,
        a % 2 == 0,
        (SUM % 2 + PROD % 3) % 2 == 0,
    ]

def run_lengths(arr: np.ndarray) -> np.ndarray:
    """Return the run lengths of a 1D array."""
    where_arr_flips = np.flatnonzero(arr[1:] != arr[:-1])
    return np.diff(where_arr_flips, prepend=-1, append=len(arr) - 1)

def run_length_penalty(symbol: Symbol) -> int:
    penalty = 0
    for row in symbol:
        runs = run_lengths(row)
        penalty += (runs[runs >= 5] - 5 + N_1).sum()
    for col in symbol.T:
        runs = run_lengths(col)
        penalty += (runs[runs >= 5] - 5 + N_1).sum()
    return penalty

def block_penalty(symbol: Symbol) -> int:
    top_left = symbol[:-1, :-1]
    top_right = symbol[:-1, 1:]
    bottom_left = symbol[1:, :-1]
    bottom_right = symbol[1:, 1:]
    return N_2 * np.sum(
        (top_left == top_right)
        & (top_left == bottom_left)
        & (top_left == bottom_right)
    )

def bad_pattern_penalty(symbol: Symbol) -> int:
    for flipped in (symbol, symbol.T):
        for rotations in range(4):
            s = np.rot90(flipped, rotations)
            # Find patterns of 10111010000
            bad_pattern = (
                s[:-10]
                & ~s[1:-9]
                &  s[2:-8]
                &  s[3:-7]
                &  s[4:-6]
                & ~s[5:-5]
                &  s[6:-4]
                & ~s[7:-3]
                & ~s[8:-2]
                & ~s[9:-1]
                & ~s[10:]
            )
            if bad_pattern.any():
                return N_3
    return 0

def proportion_penalty(symbol: Symbol) -> int:
    amount_dark = symbol.sum()
    half = symbol.size // 2
    five_percent = symbol.size // 20
    return N_4 * (abs(amount_dark - half) // five_percent)

def penalty_score(symbol: Symbol) -> int:
    return (
        run_length_penalty(symbol)
        + block_penalty(symbol)
        + bad_pattern_penalty(symbol)
        + proportion_penalty(symbol)
    )

def apply_mask(symbol: Symbol, version: int, fixed_id: int | None = None) -> tuple[Symbol, int]:
    """Find and apply the mask with the lowest penalty.

    This should be done after applying the finder, timing and alignment
    patterns, but before the format and version info.
    """
    patterns = pattern_masks(version)
    data_mask = get_data_mask(version)
    if fixed_id is None:
        # Get the symbol and pattern number that has the lowest penalty score
        masked_symbol, mask_pattern_id = min(
            ((symbol ^ (pattern & data_mask), i) for i, pattern in enumerate(patterns)),
            key=lambda pair: penalty_score(pair[0]),
        )
    else:
        masked_symbol = symbol ^ (patterns[fixed_id] & data_mask)
        mask_pattern_id = fixed_id
    return masked_symbol, mask_pattern_id
