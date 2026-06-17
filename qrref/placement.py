import numpy as np

from qrref.custom_types import *
from qrref.table_data import modules_per_side


# Distance from the last to second-to-last alignment patterns. The first and last
# are fixed, and all but the first are evenly spaced with this spacing.
# See Table E.1, page 83.
ALIGNMENT_PATTERN_SPACING: dict[int, set[int]] = {
    12: {2},
    16: {3, 7},
    18: {8},
    20: {4, 9, 14},
    22: {10, 15, 21},
    24: {5, 11, 16, 17, 22, 23, 28, 29, 35},
    26: {12, 18, 24, 25, 30, 31, 32, 36, 37, 38},
    28: {6, 13, 19, 20, 26, 27, 33, 34, 39, 40},
}
FINDER_PATTERN = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
], dtype=np.bool)
ALIGNMENT_PATTERN = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1],
], dtype=np.bool)


def alignment_pattern_axis_positions(version: int) -> list[int]:
    """Return the positions of alignment pattern centers along each axis.

    Implements Table E.1, page 83.
    """
    if version == 1:
        return []
    # Number of alignment patterns per axis
    num_per_axis = 2 + version // 7
    # Position of last pattern (in each axis)
    last = 4 * version + 10
    # Find the spacing of this version
    for spacing, versions in ALIGNMENT_PATTERN_SPACING.items():
        if version in versions:
            break
    else:
        raise ValueError("Version not found in ALIGNMENT_PATTERN_SPACING")

    # The first is always at 6, the rest are evenly spaced
    axis_positions = [6] + [last - spacing * i for i in range(num_per_axis - 2, -1, -1)]
    return axis_positions

def alignment_pattern_coordinates(version: int) -> Generator[tuple[int, int]]:
    axis_positions = alignment_pattern_axis_positions(version)
    # Top row between finders
    for col in axis_positions[1:-1]:
        yield (axis_positions[0], col)
    # Left column between finders
    for row in axis_positions[1:-1]:
        yield (row, axis_positions[0])
    # The remaining lower-right corner
    for row in axis_positions[1:]:
        for col in axis_positions[1:]:
            yield (row, col)


def blank_symbol(version: int) -> Symbol:
    """Return a Symbol containing all zeros."""
    size = modules_per_side(version)
    return np.zeros((size, size), dtype=np.bool)

def get_function_pattern_mask(version: int) -> Symbol:
    """Create a mask where 1's represent function pattern areas."""
    mask = blank_symbol(version)

    # Place finder patterns
    mask[:8, :8] = 1
    mask[:8, -8:] = 1
    mask[-8:, :8] = 1
    # Place timing patterns
    mask[6, :] = 1
    mask[:, 6] = 1
    # Place alignment patterns
    for row, col in alignment_pattern_coordinates(version):
        mask[row-2:row+3, col-2:col+3] = 1

    return mask

def get_format_and_version_pattern_mask(version: int) -> Symbol:
    """Create a mask where 1's represent format and version pattern areas."""
    mask = blank_symbol(version)

    # Place format info
    mask[8, :9] = 1
    mask[8, -8:] = 1
    mask[:9, 8] = 1
    mask[-8:, 8] = 1
    # Remove modules that belong to the timing patterns
    mask[8, 6] = 0
    mask[6, 8] = 0

    # Place version info
    if version >= 7:
        mask[:6, -11:-8] = 1
        mask[-11:-8, :6] = 1

    return mask

def get_data_mask(version: int) -> Symbol:
    """Create a mask where 1's represent data areas."""
    return ~(
        get_function_pattern_mask(version)
        | get_format_and_version_pattern_mask(version)
    )

def data_bit_positions(version: int) -> Generator[tuple[int, int]]:
    """Generate the positions of the data bytes in the order to place them in.

    Implements section 7.7 according to the procedure outlined on page 48.
    """
    # Positions to place data bits in
    data_mask = get_data_mask(version)
    # Start from the bottom-right
    width = modules_per_side(version)
    row = col = width - 1

    row_direction = -1  # Up
    while col >= 1:
        # Place bits right, then left
        if data_mask[row, col]:
            yield (row, col)
        if data_mask[row, col - 1]:
            yield (row, col - 1)
        # Move up or down
        row += row_direction
        # Change direction
        if row < 0 or row >= width:
            row -= row_direction  # Undo the row move
            col -= 2
            # Skip the column where there is a timing pattern
            if col == 6:
                col -= 1
            row_direction = -row_direction

def place_bytestream(bytestream: Generator[Byte], version: int) -> Symbol:
    symbol = blank_symbol(version)
    bit_positions = data_bit_positions(version)

    for byte in bytestream:
        for bit in to_bits(byte, 8):
            try:
                row, col = next(bit_positions)
            except StopIteration:
                raise ValueError("The bit position stream ended prematurely")
            symbol[row, col] = bit

    return symbol

def insert_finder_patterns(symbol: Symbol):
    """Insert finder patterns in all corners except the lower right."""
    symbol[:7, :7] = FINDER_PATTERN
    symbol[:7, -7:] = FINDER_PATTERN
    symbol[-7:, :7] = FINDER_PATTERN

def insert_timing_patterns(symbol: Symbol):
    """Insert timing patterns."""
    symbol[6, 8:-8:2] = 1
    symbol[8:-8:2, 6] = 1

def insert_alignment_patterns(symbol: Symbol, version: int):
    for row, col in alignment_pattern_coordinates(version):
        symbol[row - 2 : row + 3, col - 2 : col + 3] = ALIGNMENT_PATTERN


def place_format_bits(symbol: Symbol, bitstream: Generator[Bit]):
    """Place the format bits according to section 7.9, page 55.

    Each bit is placed in two positions for redundancy.
    """
    # Place bits 14-9
    for i in range(6):
        symbol[8, i] = symbol[-1 - i, 8] = next(bitstream)
    # Place bits 8-6
    symbol[8, 7] = symbol[-7, 8] = next(bitstream)
    symbol[8, 8] = symbol[8, -8] = next(bitstream)
    symbol[7, 8] = symbol[8, -7] = next(bitstream)
    # Place the dark module that is always at (4*version+9, 8)
    symbol[-8, 8] = 1
    # Place bits 6-0
    for i in range(6):
        symbol[5 - i, 8] = symbol[8, -6 + i] = next(bitstream)

def place_version_bits(symbol: Symbol, bitstream: Generator[Bit]):
    """Place the version bits according to section 7.10, page 58.

    Each bit is placed in two positions for redundancy.
    """
    for i in range(5, -1, -1):
        for j in range(-9, -12, -1):
            symbol[i, j] = symbol[j, i] = next(bitstream)

def expand_quiet_region(symbol: Symbol, width: int = 4) -> Symbol:
    rows, cols = symbol.shape
    new_symbol = np.zeros((rows + 2 * width, cols + 2 * width), dtype=np.bool)
    new_symbol[width:-width, width:-width] = symbol
    return new_symbol
