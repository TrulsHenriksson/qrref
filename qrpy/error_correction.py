import numpy as np

import functools

from qrpy.custom_types import *
from qrpy.table_data import BLOCK_TYPES
from qrpy.galois_field import GFE


# Constant mask that guarantees the format bit string is never all zeros
FORMAT_MASK = 0b101010000010010
# Represents x^10 + x^8 + x^5 + x^4 + x^2 + x + 1
FORMAT_GENERATOR_POLYNOMIAL = 0b10100110111
# Represents x^12 + x^11 + x^10 + x^9 + x^8 + x^5 + x^2 + 1
VERSION_GENERATOR_POLYNOMIAL = 0b1111100100101


@functools.cache
def generating_polynomial(order: int) -> list[GFE]:
    """Get the generating polynomial of a given order.
    
    Cached (memoized) for speed and reusability.

    Polynomials are here given by their coefficients, ordered from lowest to
    highest power. So `1 + 3x + 2x^3` would be `[1, 3, 0, 2]`.
    """
    if order <= 0:
        raise ValueError("order must be at least 1")
    if order == 1:
        one = GFE(1)
        return [one, one]  # $(\alpha^0 + x)$

    polynomial = generating_polynomial(order - 1)
    zero = GFE(0)
    alpha = GFE.exp(1)  # Element of order 255

    # Multiply polynomial by $(\alpha^i + x)$
    factor = alpha**(order - 1)
    times_factor = [factor * c for c in polynomial] + [zero]
    times_x = [zero] + polynomial
    polynomial = [c1 + c2 for c1, c2 in zip(times_factor, times_x)]
    return polynomial


def polynomial_remainder[T: FieldElement](a: Sequence[T], g: Sequence[T]) -> list[T]:
    """Calculate the remainder after polynomial division of `a` by `g`."""
    # Easier to work with in highest-exponent-first order
    result = list(a)[::-1]
    g = g[::-1]
    for i in range(len(a) - len(g) + 1):
        c = result[i] / g[0]
        # Subtract $c \cdot x^i g$ from result
        for j in range(len(g)):
            result[i + j] -= c * g[j]
    # Return in lowest-exponent-first order
    return result[-1 : -len(g) : -1]


def generate_error_correction_blocks(
    bytestream: Generator[Byte], version: int, ec_level: ErrorCorrectionLevel
) -> Generator[tuple[list[Byte], list[Byte]]]:
    """Divide the bytestream into blocks and add error correction bytes to each block.

    `bytestream` should come from `data_encoding.to_data_bytestream`, and is assumed to
    be infinite.

    Each block of data bytes gets a block of error correction bytes. These are
    generated in parallel. The size and number of these blocks are defined by
    Table 9, page 38-44.

    Implements section 7.6, points 1 and 2 on page 45.
    """
    block_types = BLOCK_TYPES[ec_level][version]
    for num_blocks, c, k in block_types:
        for i in range(num_blocks):
            # Grab the next k bytes
            data_block = [next(bytestream) for i in range(k)]

            # Coefficients of the data polynomial
            coefficients = [GFE(0)] * (c - k) + [GFE(byte) for byte in reversed(data_block)]
            g = generating_polynomial(c - k)
            ec_coefficients = polynomial_remainder(coefficients, g)

            # Convert to error correction bytes
            ec_block = [c.value for c in reversed(ec_coefficients)]
            yield data_block, ec_block


def interleave_blocks(
    blockstream: Iterable[tuple[list[Byte], list[Byte]]],
    version: int,
    ec_level: ErrorCorrectionLevel,
) -> Generator[Byte]:
    """Create the final bytestream from the data and error correction blockstreams.

    `blockstream` should come from `generate_error_correction_blocks`.

    Implements section 7.6, point 3 on page 45, according to Figure 15, page 46.
    """
    # Each ec code has one or two block types given by tuples (n, c, k, r) (see Table 9)
    block_types = BLOCK_TYPES[ec_level][version]
    longest_data_block = max(k for _, _, k in block_types)
    # All ec blocks have the same length so this `max` isn't strictly necessary
    longest_ec_block = max(c - k for _, c, k in block_types)
    total_num_blocks = sum(num_blocks for num_blocks, _, _ in block_types)

    # Make room for the final blocks (-1 signals missing values due to different block lengths)
    data_blocks = np.full((total_num_blocks, longest_data_block), -1, dtype=np.int16)
    ec_blocks = np.full((total_num_blocks, longest_ec_block), -1, dtype=np.int16)

    # Populate the matrices
    for i, (data_block, ec_block) in enumerate(blockstream):
        data_blocks[i, :len(data_block)] = data_block
        ec_blocks[i, :len(ec_block)] = ec_block

    # Illustrating example for the next section:
    # >>> a = np.array([[1,  2, -1, -1],
    # ...               [3,  4,  5, -1],
    # ...               [6,  7,  8,  9]])
    # >>> a.T[a.T != -1]
    # array([1, 3, 6, 2, 4, 7, 5, 8, 9])

    # Remove missing data bytes and flatten, vertically first
    data_bytes = data_blocks.T[data_blocks.T != -1]
    yield from data_bytes
    ec_bytes = ec_blocks.T[ec_blocks.T != -1]
    yield from ec_bytes


def generate_format_bits(ec_level: ErrorCorrectionLevel, mask_pattern_id: int) -> Generator[Bit]:
    """Generate the data and error correction bits for the format information.

    Uses a BCH (15, 5) code to generate error correction bits. This means the
    data is 5 bits long, and the error correction bits are the coefficients of
    the data polynomial times x^(15-5), mod the generating polynomial of degree
    10. We thus get 5 data bits and 10 error correction bits.

    Implements Annex C.2, page 79.
    """
    ec_indicator = {"L": 0b01, "M": 0b00, "Q": 0b11, "H": 0b10}[ec_level]
    # The data bits are 0bAABBB where AA is ec_indicator and BBB is mask_pattern_id
    data_bits = (ec_indicator << 3) | mask_pattern_id

    # Polynomial with binary coefficients, multiplied by x^10 (so now 15 bits)
    remainder = data_bits << 10
    # Perform binary polynomial division by the generating polynomial
    divisor = FORMAT_GENERATOR_POLYNOMIAL << (5 - 1)
    highest_bit = 1 << (15 - 1)
    for i in range(5):
        if remainder & highest_bit:
            remainder ^= divisor
        divisor >>= 1
        highest_bit >>= 1

    # Concatenate the data bits with the polynomial remainder
    bits = (data_bits << 10) | remainder
    yield from to_bits(bits ^ FORMAT_MASK, 15)

def generate_version_bits(version: int) -> Generator[Bit]:
    """Generate the data and error correction bits for the version number.

    Uses the Golay (18, 6) code to generate error correction bits. This means the
    data is 6 bits long, and the error correction bits are the coefficients of
    the data polynomial times x^(18-6), mod the generating polynomial of degree
    12. We thus get 6 data bits and 12 error correction bits.

    Implements Annex D.2, page 81.
    """
    # Polynomial with binary coefficients, multiplied by x^12 (so now 18 bits)
    remainder = version << 12
    # Perform binary polynomial division by the generating polynomial
    divisor = VERSION_GENERATOR_POLYNOMIAL << (6 - 1)
    highest_bit = 1 << (18 - 1)
    for i in range(6):
        if remainder & highest_bit:
            remainder ^= divisor
        divisor >>= 1
        highest_bit >>= 1

    yield from to_bits(version, 6)
    yield from to_bits(remainder, 12)
