import numpy as np

from typing import Generator, Sequence, Literal, Iterable, Callable, Protocol


__all__ = [
    "Bit",
    "Byte",
    "Mode",
    "ErrorCorrectionLevel",
    "Symbol",
    "FieldElement",
    "Generator",
    "Sequence",
    "Literal",
    "Iterable",
    "Callable",
    "to_bits",
    "to_byte",
]


class FieldElement[T](Protocol):
    def __add__(self: T, other: T, /) -> T: ...
    def __sub__(self: T, other: T, /) -> T: ...
    def __mul__(self: T, other: T, /) -> T: ...
    def __truediv__(self: T, other: T, /) -> T: ...


type Bit = Literal[0, 1]
type Byte = int  # 0-255
type Mode = Literal["numeric", "alphanumeric", "byte"]
type ErrorCorrectionLevel = Literal["L", "M", "Q", "H"]
type Symbol = np.ndarray[tuple[int, int], np.dtype[np.bool]]


def to_bits(value: int, length: int) -> Generator[Bit]:
    """Generate the lowest `length` bits of `value`, highest bit first."""
    if length < 1:
        raise ValueError("length must be positive")
    bit_position = 1 << length - 1
    while bit_position:
        yield 1 if value & bit_position else 0
        bit_position >>= 1

def to_byte(bits: tuple[Bit, Bit, Bit, Bit, Bit, Bit, Bit, Bit]) -> Byte:
    """Convert a tuple of 8 bits to its numeric value as a byte (uint8)."""
    byte = 0
    for bit in bits:
        byte = (byte << 1) + bit
    return byte
