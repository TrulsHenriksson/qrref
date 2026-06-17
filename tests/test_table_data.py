import pytest


from qrref.table_data import (
    BLOCK_TYPES,
    modules_per_side,
    function_pattern_modules,
    format_and_version_modules,
    data_modules,
)


def test_data_capacity():
    """The block size times the number of blocks should equal the number of bytes that fit."""
    for level in ("L", "M", "Q", "H"):
        for version in range(1, 41):
            block_types = BLOCK_TYPES[level][version]
            data_bytes = sum(num_blocks * c for num_blocks, c, _ in block_types)
            assert data_bytes == data_modules(version) // 8, f"{level = }, {version = } failed"

def test_block_order():
    """The shortest error correction blocks should come first."""
    for level in ("L", "M", "Q", "H"):
        for version in range(1, 41):
            block_types = BLOCK_TYPES[level][version]
            if len(block_types) == 2:
                (_, c1, k1), (_, c2, k2) = block_types
                assert c1 == c2 - 1, f"{level = }, {version = } failed"
                assert k1 == k2 - 1, f"{level = }, {version = } failed"


def test_modules_per_side():
    # From pictures on pages 10-15
    assert modules_per_side(1) == 21
    assert modules_per_side(2) == 25
    assert modules_per_side(6) == 41
    assert modules_per_side(7) == 45
    assert modules_per_side(14) == 73
    assert modules_per_side(21) == 101
    assert modules_per_side(40) == 177

def test_function_pattern_modules():
    # Table 1, page 19-20, column B
    assert function_pattern_modules(1) == 202
    assert function_pattern_modules(2) == 235
    assert function_pattern_modules(6) == 267
    assert function_pattern_modules(7) == 390
    assert function_pattern_modules(14) == 611
    assert function_pattern_modules(21) == 882
    assert function_pattern_modules(40) == 1614

def test_format_and_version_modules():
    # Table 1, page 19-20, column C
    assert format_and_version_modules(1) == 31
    assert format_and_version_modules(2) == 31
    assert format_and_version_modules(6) == 31
    assert format_and_version_modules(7) == 67
    assert format_and_version_modules(14) == 67
    assert format_and_version_modules(21) == 67
    assert format_and_version_modules(40) == 67

def test_data_modules():
    # Table 1, page 19-20, column D
    assert data_modules(1) == 208
    assert data_modules(2) == 359
    assert data_modules(6) == 1383
    assert data_modules(7) == 1568
    assert data_modules(14) == 4651
    assert data_modules(21) == 9252
    assert data_modules(40) == 29648
