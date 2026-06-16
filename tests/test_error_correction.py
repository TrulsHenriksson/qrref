import pytest

from qrpy.custom_types import ErrorCorrectionLevel, Mode, to_bits
from qrpy.data_encoding import encode_alphanumeric, encode_byte, encode_numeric, to_data_bytestream
from qrpy.error_correction import (
    generate_version_bits,
    generating_polynomial,
    polynomial_remainder,
    generate_error_correction_blocks,
    interleave_blocks,
    generate_format_bits,
)
from qrpy.galois_field import GFE
from qrpy.table_data import BLOCK_TYPES, value_of_p


def test_generating_polynomial():
    # Table A.1, page 73
    assert [c.log() for c in generating_polynomial(2)] == [
        0, 25, 1
    ]
    assert [c.log() for c in generating_polynomial(7)] == [
        0, 87, 229, 146, 149, 238, 102, 21
    ]
    # The 68th generating polynomial is $\alpha^0 x^{68} + \alpha^{247}x^{67} + \cdots$
    assert [c.log() for c in generating_polynomial(68)] == [
        0, 247, 159, 223, 33, 224, 93, 77, 70, 90, 160, 32, 254, 43, 150, 84,
        101, 190, 205, 133, 52, 60, 202, 165, 220, 203, 151, 93, 84, 15, 84,
        253, 173, 160, 89, 227, 52, 199, 97, 95, 231, 52, 177, 41, 125, 137,
        241, 166, 225, 118, 2, 54, 32, 82, 215, 175, 198, 43, 238, 235, 27,
        101, 184, 127, 3, 5, 8, 163, 238
    ]

def test_polynomial_remainder():
    a = [16.0, 36.0, 76.0, 87.0, 66.0, 43.0, 10.0]
    g = [ 2.0,  3.0,  6.0,  4.0]
    # Long division steps:
    #    16    36    76    87    66    43    10
    #     0    12    28    55    66    43    10      -8x^3 * g
    #           0    10    19    42    43    10      -6x^2 * g
    #                 0     4    12    23    10      -5x   * g
    #                       0     6    11     2      -2    * g
    assert polynomial_remainder(a, g) == [6.0, 11.0, 2.0]

def test_blockstream_length():
    for ec_level in ("L",):  # Others omitted for time reasons
        for version in range(1, 41):
            bytestream = to_data_bytestream((1 for i in range(152)), version, ec_level)
            blockstream = generate_error_correction_blocks(bytestream, version, ec_level)

            for (num_blocks, c, k) in BLOCK_TYPES[ec_level][version]:
                # Make sure the blockstream consists of n blocks of lengths k, c-k respectively
                for i in range(num_blocks):
                    data_block, ec_block = next(blockstream)
                    assert len(data_block) == k
                    assert len(ec_block) == c - k
            # Make sure the blockstream is depleted
            with pytest.raises(StopIteration):
                next(blockstream)

def test_syndromes():
    """Make sure the syndromes (detected errors) of a sample message are zero.

    Follows the example in Annex B, page 77.
    """
    version = 1
    ec_level = "M"
    # We get p=2, c=26, k=16 in this case
    p = value_of_p(version, ec_level)
    (_, c, k), *_ = BLOCK_TYPES[ec_level][version]

    sample_message = to_data_bytestream((1 for i in range(128)), version, ec_level)
    blockstream = generate_error_correction_blocks(sample_message, version, ec_level)

    for data_block, ec_block in blockstream:  # Only one pair of blocks in this case
        full_message = data_block + ec_block
        # Full message (contains c=26 bytes) interpreted as a polynomial over GF256
        polynomial = [GFE(byte) for byte in reversed(full_message)]

        # Calculate n = c-k-p syndromes
        for i in range(c - k - p):
            alpha_i = GFE.exp(i)
            # Evaluate the polynomial at $\alpha^i$
            syndrome = sum((coeff * alpha_i**j for j, coeff in enumerate(polynomial)), start=GFE(0))
            assert syndrome.value == 0


def test_generate_format_bits():
    # Make sure the result matches Table C.1, page 80
    correct_hex_codes = [
        0x5412, 0x5125, 0x5E7C, 0x5B4B, 0x45F9, 0x40CE, 0x4F97, 0x4AA0,
        0x77C4, 0x72F3, 0x7DAA, 0x789D, 0x662F, 0x6318, 0x6C41, 0x6976,
        0x1689, 0x13BE, 0x1CE7, 0x19D0, 0x0762, 0x0255, 0x0D0C, 0x083B,
        0x355F, 0x3068, 0x3F31, 0x3A06, 0x24B4, 0x2183, 0x2EDA, 0x2BED,
    ]
    for i, ec_level in enumerate(("M", "L", "H", "Q")):
        for j in range(8):
            data_bits = (i << 3) | j
            assert (
                list(generate_format_bits(ec_level, j))
                == list(to_bits(correct_hex_codes[data_bits], 15))
            ), f"{i = }, {j = } failed"

def test_generate_version_bits():
    # Make sure the result matches Table D.1, page 82
    correct_hex_codes = [
        0x07C94, 0x085BC, 0x09A99, 0x0A4D3, 0x0BBF6, 0x0C762, 0x0D847, 0x0E60D,
        0x0F928, 0x10B78, 0x1145D, 0x12A17, 0x13532, 0x149A6, 0x15683, 0x168C9,
        0x177EC, 0x18EC4, 0x191E1, 0x1AFAB, 0x1B08E, 0x1CC1A, 0x1D33F, 0x1ED75,
        0x1F250, 0x209D5, 0x216F0, 0x228BA, 0x2379F, 0x24B0B, 0x2542E, 0x26A64,
        0x27541, 0x28C69,
    ]
    for version, hex_code in zip(range(7, 41), correct_hex_codes):
        assert (
            list(generate_version_bits(version))
            == list(to_bits(hex_code, 18))
        ), f"{version = } failed"


# Part of Table 7, page 33-36
DATA_CAPACITIES: dict[int, dict[ErrorCorrectionLevel, dict[Mode, int]]] = {
    1: {
        "L": {"numeric": 41, "alphanumeric": 25, "byte": 17},
        "M": {"numeric": 34, "alphanumeric": 20, "byte": 14},
        "Q": {"numeric": 27, "alphanumeric": 16, "byte": 11},
        "H": {"numeric": 17, "alphanumeric": 10, "byte": 7},
    },
    7: {
        "L": {"numeric": 370, "alphanumeric": 224, "byte": 154},
        "M": {"numeric": 293, "alphanumeric": 178, "byte": 122},
        "Q": {"numeric": 207, "alphanumeric": 125, "byte": 86},
        "H": {"numeric": 154, "alphanumeric": 93, "byte": 64},
    },
    40: {
        "L": {"numeric": 7089, "alphanumeric": 4296, "byte": 2953},
        "M": {"numeric": 5596, "alphanumeric": 3391, "byte": 2331},
        "Q": {"numeric": 3993, "alphanumeric": 2420, "byte": 1663},
        "H": {"numeric": 3057, "alphanumeric": 1852, "byte": 1273},
    },
}
ENCODERS = {
    "numeric": encode_numeric, "alphanumeric": encode_alphanumeric, "byte": encode_byte
}


# Parameterized to be able to locate which ones fail
@pytest.mark.parametrize("ec_level", ["L", "M", "Q", "H"])
@pytest.mark.parametrize("version", [1, 7, 40])
@pytest.mark.parametrize("mode", ["numeric", "alphanumeric", "byte"])
def test_bytestream_data_capacity(version: int, ec_level: ErrorCorrectionLevel, mode: Mode):
    encoder = ENCODERS[mode]
    capacity = DATA_CAPACITIES[version][ec_level][mode]
    
    # Make sure you can take all values
    bitstream = encoder("0" * capacity, version)
    list(generate_error_correction_blocks(
        to_data_bytestream(bitstream, version, ec_level),
            version,
            ec_level,
    ))

    # Make sure the capacity can be exceeded
    bitstream = encoder("0" * (capacity + 1), version)
    with pytest.raises(ValueError):
        list(generate_error_correction_blocks(
            to_data_bytestream(bitstream, version, ec_level),
                version,
                ec_level,
        ))
