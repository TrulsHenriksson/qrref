from qrpy.custom_types import *
from qrpy.table_data import BLOCK_TYPES
from qrpy.data_encoding import (
    NUMERIC_CHARS,
    ALPHANUMERIC_EXCLUSIVE_CHARS,
    BYTE_CHARS,
    BYTE_EXCLUSIVE_CHARS,
)


def select_modes(content: str, version: int):
    """Select the modes that generates the shortest bitstream.

    Implements section J.2, page 100.
    """
    def select(a: int, b: int, c: int) -> int:
        return a if version <= 9 else b if version <= 26 else c

    # Select initial mode according to J.2.a)
    current_char = content[0]
    if current_char in BYTE_EXCLUSIVE_CHARS:
        current_mode = "byte"
    elif current_char in ALPHANUMERIC_EXCLUSIVE_CHARS:
        # Start with byte mode if any of the first 6/7/8 chars require it
        if needs_mode(content[: select(6, 7, 8)], BYTE_EXCLUSIVE_CHARS):
            current_mode = "byte"
        else:
            current_mode = "alphanumeric"
    elif current_char in NUMERIC_CHARS:
        # Start with byte mode if any of the first 4/4/5 chars require it
        if needs_mode(content[: select(4, 4, 5)], BYTE_EXCLUSIVE_CHARS):
            current_mode = "byte"
        # Start with alphanumeric mode if any of the first 7/8/9 chars require it
        elif needs_mode(content[: select(7, 8, 9)], ALPHANUMERIC_EXCLUSIVE_CHARS):
            current_mode = "alphanumeric"
        else:
            current_mode = "numeric"
    else:
        raise ValueError("First character of content is not encodeable")

    start_index = 0
    modes: list[tuple[str, Mode]] = []
    next_mode: Mode = current_mode
    # Select mode while moving through the content
    for i, current_char in enumerate(content[1:], start=1):
        if current_char not in BYTE_CHARS:
            raise ValueError(f"Character {current_char} at position {i} is not encodeable")

        # Update the next mode
        match current_mode:
            case "byte":
                # Switch to alphanumeric if the next 11/15/16 chars are worth it
                if worth_switching(content, ALPHANUMERIC_EXCLUSIVE_CHARS, i, select(11, 15, 16)):
                    next_mode = "alphanumeric"
                elif worth_switching(content, NUMERIC_CHARS, i, select(6, 8, 9)):
                    next_mode = "numeric"
            case "alphanumeric":
                if current_char in BYTE_EXCLUSIVE_CHARS:
                    next_mode = "byte"
                elif worth_switching(content, NUMERIC_CHARS, i, select(13, 15, 17)):
                    next_mode = "numeric"
            case "numeric":
                if current_char in BYTE_EXCLUSIVE_CHARS:
                    next_mode = "byte"
                elif current_char in ALPHANUMERIC_EXCLUSIVE_CHARS:
                    next_mode = "alphanumeric"

        # Update the running list of modes
        if next_mode != current_mode:
            modes.append((content[start_index:i], current_mode))
            start_index = i
            current_mode = next_mode

    # Add the final segment
    modes.append((content[start_index:], current_mode))
    return modes

def needs_mode(content: str, charset: set[str]):
    """Return whether `content` contains characters in `charset`."""
    return any(char in charset for char in content)

def worth_switching(content: str, charset: set[str], i: int, length: int):
    """Return whether the first `length` characters in `content[i:]` are all in `charset`."""
    return len(content) - i >= length and all(
        char in charset for char in content[i : i + length]
    )


def mixed_encoding_length(modes: list[tuple[str, Mode]], version: int) -> int:
    """Return the length of the data bitstream as returned by `encoding_mixed`."""
    def select(a: int, b: int, c: int) -> int:
        return a if version <= 9 else b if version <= 26 else c

    length = 0
    for content, mode in modes:
        match mode:
            case "numeric":
                length += 4 + select(10, 12, 14) + (len(content) // 3) * 10
                length += (0, 4, 7)[len(content) % 3]
            case "alphanumeric":
                length += 4 + select(9, 11, 13) + (len(content) // 2) * 11
                length += (0, 6)[len(content) % 2]
            case "byte":
                length += 4 + select(8, 16, 16) + len(content) * 8
            case _:
                raise ValueError(f"Invalid mode {mode}")
    return length


def integer_secant_method(func: Callable[[int], int], a: int, b: int):
    """Find the lowest integer in `{a, a+1, ..., b}` that gives a nonnegative value for `func`.

    `func` is assumed to be strictly increasing. Otherwise we would have to use
    the bisection method.
    """
    f_a = func(a)
    if f_a >= 0:
        return a
    f_b = func(b)
    if f_b == 0:
        return b

    while a + 1 < b:
        # Take a guess in the middle from the secant method
        mid = max(a + 1, min(b - 1, int(b - (b - a) * f_b / (f_b - f_a))))
        f_mid = func(mid)
        if f_mid == 0:
            return mid
        elif f_mid < 0:
            a, f_a = mid, f_mid
        else:
            b, f_b = mid, f_mid

    return b


def select_version(content: str, ec_level: ErrorCorrectionLevel):
    """Find the lowest version that can contain `content` with the given `ec_level`."""

    # Test three ranges of versions, since the encoding only changes between 9/10 and 26/27
    for a, b in ((1, 9), (10, 26), (27, 40)):
        modes = select_modes(content, version=a)
        bitstream_length = mixed_encoding_length(modes, version=a)

        # Number of data bits that fits a symbol of a given version
        symbol_capacity = lambda version: 8 * sum(
            num_blocks * k for num_blocks, _, k in BLOCK_TYPES[ec_level][version]
        )
        unused_modules = lambda version: symbol_capacity(version) - bitstream_length

        if unused_modules(b) >= 0:
            break
    else:
        raise ValueError(
            "The content doesn't fit into any symbol with error correction level"
            f" {ec_level}"
        )

    # We know unused_modules is positive at b, now find the smallest version <= b that fits
    version = integer_secant_method(unused_modules, a, b)
    return version, modes
