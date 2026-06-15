import itertools as it

from qrpy.custom_types import *
from qrpy.settings import SETTINGS
from qrpy.table_data import BLOCK_TYPES


NUMERIC_CHARS = set("0123456789")

# Numeric values of characters in alphanumeric mode
ALPHANUMERIC_VALUE = {
    char: i
    for i, char in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:")
}
ALPHANUMERIC_CHARS = set(ALPHANUMERIC_VALUE)
ALPHANUMERIC_EXCLUSIVE_CHARS = ALPHANUMERIC_CHARS - NUMERIC_CHARS

LATIN_1_CHARS = set(bytes(range(256)).decode("latin-1"))
LATIN_1_EXCLUSIVE_CHARS = LATIN_1_CHARS - ALPHANUMERIC_CHARS

# Value to signal the end of a message stream
TERMINATOR = (0, 0, 0, 0)
# Bytes to pad with at the end of the data stream
PADDING_BYTES: list[Byte] = [0b11101100, 0b00010001]


def encode_numeric(content: str, version: int) -> Generator[Bit]:
    """Encode a string of numeric characters as a bit stream."""
    # Mode indicator: 0001 for numeric mode
    yield from (0, 0, 0, 1)

    # Character count indicator
    if version <= 9:  # Version 1-9
        yield from to_bits(len(content), 10)
    elif version <= 26:  # Version 10-26
        yield from to_bits(len(content), 12)
    else:  # Version 27-40
        yield from to_bits(len(content), 14)

    # Data bitstream
    for i in range(0, len(content), 3):
        chars = content[i : i + 3]
        if len(chars) == 3:
            # Three digits are encoded with 10 bits since 2^10 = 1024 > 10^3 = 1000
            yield from to_bits(int(chars), 10)
        elif len(chars) == 2:
            # Two trailing digits can be encoded with 7 bits
            yield from to_bits(int(chars), 7)
        else:
            # One trailing digit can be encoded with 4 bits
            yield from to_bits(int(chars), 4)


def encode_alphanumeric(content: str, version: int) -> Generator[Bit]:
    """Encode a string of alphanumeric characters as a bit stream."""
    # Mode indicator: 0010 for alphanumeric mode
    yield from (0, 0, 1, 0)

    # Character count indicator
    if version <= 9:  # Version 1-9
        yield from to_bits(len(content), 9)
    elif version <= 26:  # Version 10-26
        yield from to_bits(len(content), 11)
    else:  # Version 27-40
        yield from to_bits(len(content), 13)

    # Data bitstream
    for i in range(0, len(content), 2):
        chars = content[i : i + 2]
        if len(chars) == 2:
            # A pair of characters are encoded with 11 bits since 2^11 = 2048 > 45^2 = 2025
            first, second = chars
            yield from to_bits(
                45 * ALPHANUMERIC_VALUE[first] + ALPHANUMERIC_VALUE[second], 11
            )
        else:
            # A trailing single character is encoded with 6 bits
            (first,) = chars
            yield from to_bits(ALPHANUMERIC_VALUE[first], 6)


def encode_byte(content: str, version: int) -> Generator[Bit]:
    """Encode a string of bytes as a bit stream."""
    # Mode indicator: 0100 for byte mode
    yield from (0, 1, 0, 0)

    # Character count indicator
    encoded_content = content.encode(SETTINGS.byte_encoding)
    if version <= 9:  # Version 1-9
        yield from to_bits(len(encoded_content), 8)
    else:  # Version 10-40
        yield from to_bits(len(encoded_content), 16)

    # Data bitstream
    for byte in encoded_content:
        yield from to_bits(byte, 8)


def encode_mixed(modes: list[tuple[str, Mode]], version: int) -> Generator[Bit]:
    """Encode a string using mixed modes."""
    for content, mode in modes:
        match mode:
            case "numeric":
                yield from encode_numeric(content, version)
            case "alphanumeric":
                yield from encode_alphanumeric(content, version)
            case "byte":
                yield from encode_byte(content, version)
            case "ECI" | "kanji" | "structured_append" | "FNC1":
                raise NotImplementedError
            case _:
                raise ValueError(f"Invalid mode {mode}")


def terminate(data_bitstream: Iterable[Bit], max_length: int) -> Generator[Bit]:
    """End a bitstream by appending the terminator (0000)."""
    data_length = 0
    for bit in data_bitstream:
        data_length += 1
        if data_length > max_length:
            raise ValueError("Data length exceeded the capacity of the symbol")
        yield bit

    # Yield the terminator, truncated to the max length
    terminator_length = min(len(TERMINATOR), max_length - data_length)
    yield from TERMINATOR[:terminator_length]


def to_data_bytestream(
    bitstream: Iterable[Bit], version: int, ec_level: ErrorCorrectionLevel
) -> Generator[Byte]:
    """Terminate the bitstream and group the bits together into bytes.

    Use `bytestream.send(True)` when the bytestream should end to make sure all
    data bytes were included. Otherwise it will keep generating padding bytes
    indefinitely.
    """
    num_data_bytes = sum(num_blocks * k for num_blocks, _, k in BLOCK_TYPES[ec_level][version])
    terminated_bitstream = terminate(bitstream, num_data_bytes * 8)

    # Store all data bytes (to make sure the bitstream is not too long)
    data_bytes: list[Byte] = []
    try:
        for bits in it.batched(terminated_bitstream, 8):
            # The final bits might not hit the byte boundary exactly, so add zeros to align
            if len(bits) < 8:
                bits = (*bits, 0, 0, 0, 0, 0, 0, 0, 0)[:8]
            data_bytes.append(to_byte(bits))  # type: ignore
    except ValueError:
        # Re-raise with a more helpful error message
        raise ValueError(f"Too much data to fit in a {version}-{ec_level} symbol") from None

    yield from data_bytes
    # Pad before error correction bytes
    for byte in it.cycle(PADDING_BYTES):
        yield byte
