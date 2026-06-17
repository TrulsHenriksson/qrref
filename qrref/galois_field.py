import numpy as np

from qrref.custom_types import Byte


# Coefficients of $\alpha^i$ mod 2 and $x^8+x^4+x^3+x^2+1$ where $\alpha=x$
EXP = np.zeros(255, dtype=np.uint8)
EXP[0] = 0b00000001
for exponent in range(1, 255):
    EXP[exponent] = EXP[exponent - 1] << 1  # multiply by x
    if EXP[exponent - 1] & 0b10000000:  # if 1x^8...
        EXP[exponent] ^= 0b00011101  # ...subtract x^4+x^3+x^2+1

# Inverse table. Includes 0 so that `LOG[EXP[n]] == n` is constant, but LOG[0]
# should never be accessed.
LOG = np.zeros(256, dtype=np.uint8)
for exponent, value in enumerate(EXP):
    LOG[value] = exponent


class GFE:
    """Element of the Galois Field of order 256.

    An element of this field is a polynomial of degree <8 with coefficients
    mod 2 after reducing mod `x^8 + x^4 + x^3 + x^2 + 1`, for example
    `0x^7 + 1x^6 + 1x^5 + 0x^4 + 0x^3 + 1x^2 + 1x^1 + 1x^0`,
    canonically represented using its coefficients (so `0b01100111` in this case).
    Because the coefficients are mod 2, addition and subtraction are both
    just xor. Multiplication is more difficult, but made simple by precomputed
    LOG and EXP tables. Every element except zero can be written as `0b00000010`
    (representing `x`) to some power n, which is stored in `EXP[n]`. The product
    `GFE(EXP[m]) * GFE(EXP[n])` is then just `GFE(EXP[(m + n) % 255])`.
    """

    def __init__(self, value):
        self.value: Byte = int(value) & 0b11111111  # mod 256

    @staticmethod
    def exp(exponent: int) -> GFE:
        return GFE(EXP[exponent % 255])  # since $a^{255} = 1$

    def log(self) -> int:
        return int(LOG[self.value])

    def __repr__(self) -> str:
        return f"GFE({self.value})"

    def __add__(self, other: GFE) -> GFE:
        return GFE(self.value ^ other.value)

    def __sub__(self, other: GFE) -> GFE:
        return GFE(self.value ^ other.value)

    def __mul__(self, other: GFE) -> GFE:
        if self.value == 0 or other.value == 0:
            return GFE(0)
        return GFE.exp(self.log() + other.log())

    def __truediv__(self, other: GFE) -> GFE:
        if other.value == 0:
            raise ZeroDivisionError
        if self.value == 0:
            return GFE(0)
        return GFE.exp(self.log() - other.log())

    def __pow__(self, power: int) -> GFE:
        if self.value == 0:
            return GFE(0)
        return GFE.exp(self.log() * power)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GFE):
            return False
        return self.value == other.value
