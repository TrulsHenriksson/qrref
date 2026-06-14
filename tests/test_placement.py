import pytest

from qrpy.placement import (
    ALIGNMENT_PATTERN_SPACING,
    alignment_pattern_axis_positions,
    alignment_pattern_coordinates,
    get_function_pattern_mask,
    get_format_and_version_pattern_mask,
    get_data_mask,
    place_bytestream,
)
from qrpy.table_data import (
    BLOCK_TYPES,
    function_pattern_modules,
    format_and_version_modules,
    data_modules,
)

def test_alignment_pattern_spacing():
    # Make sure all versions are represented
    combined = set()
    for versions in ALIGNMENT_PATTERN_SPACING.values():
        combined |= versions
    assert combined == set(range(2, 41))

def test_alignment_pattern_axis_positions():
    # Table E.1, page 83-84
    assert alignment_pattern_axis_positions(1) == []
    assert alignment_pattern_axis_positions(2) == [6, 18]
    assert alignment_pattern_axis_positions(6) == [6, 34]
    assert alignment_pattern_axis_positions(7) == [6, 22, 38]
    assert alignment_pattern_axis_positions(14) == [6, 26, 46, 66]
    assert alignment_pattern_axis_positions(21) == [6, 28, 50, 72, 94]
    assert alignment_pattern_axis_positions(40) == [6, 30, 58, 86, 114, 142, 170]

def test_alignment_pattern_number():
    # Table E.1, page 83-84
    assert len(list(alignment_pattern_coordinates(1))) == 0
    for version in range(2, 41):
        assert len(list(alignment_pattern_coordinates(version))) == (version // 7 + 2) ** 2 - 3

def test_mask_overlap():
    for version in range(1, 41):
        mask1 = get_function_pattern_mask(version)
        mask2 = get_format_and_version_pattern_mask(version)
        mask3 = get_data_mask(version)
        # Make sure the masks don't overlap
        assert not (mask1 & mask2).any()
        assert not (mask1 & mask3).any()
        assert not (mask2 & mask3).any()
        # Make sure the masks cover the whole symbol
        assert (mask1 | mask2 | mask3).all()

def test_function_module_count():
    for version in range(1, 41):
        mask = get_function_pattern_mask(version)
        assert mask.sum() == function_pattern_modules(version)

def test_format_and_version_module_count():
    for version in range(1, 41):
        mask = get_format_and_version_pattern_mask(version)
        assert mask.sum() == format_and_version_modules(version)

def test_data_module_count():
    for version in range(1, 41):
        mask = get_data_mask(version)
        assert mask.sum() == data_modules(version)

def test_symbol_fit():
    for version in range(1, 41):
        for ec_level in ("L", "M", "Q", "H"):
            block_types = BLOCK_TYPES[ec_level][version]
            max_bytes = sum(num_blocks * c for num_blocks, c, _ in block_types)
            bytestream = (0b11111111 for i in range(max_bytes))
            # Make sure this does not raise any errors
            symbol = place_bytestream(bytestream, version)
            # Make sure the zero bits in the data region are exactly the remainder
            assert (~symbol & get_data_mask(version)).sum() == data_modules(version) % 8

def test_byte_order():
    # See Figure 5, page 10 and Figure 16, page 47
    version = 1
    bytestream = (0b10000000 for i in range(26))
    symbol = place_bytestream(bytestream, version)
    # Make sure the bytes are placed highest-bit first
    assert symbol[20, 20] == 1
    assert symbol[16, 20] == 1
    assert symbol[12, 20] == 1
    assert symbol[9, 18] == 1
    assert symbol[13, 18] == 1
    assert symbol[17, 18] == 1
