import pytest

from qrpy.data_encoding import encode_mixed
from qrpy.data_analysis import (
    select_modes,
    mixed_encoding_length,
    integer_secant_method,
)

@pytest.mark.parametrize("message", [
    "123456789",
    "1234567891",
    "12345678912",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZA",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZAB",
    "abcdefghijklmnopqrstuvwxyz",
    "abcdefghijklmnopqrstuvwxyza",
    "abcdefghijklmnopqrstuvwxyzab",
])
def test_mixed_encoding_length(message):
    for version in (9, 26, 40):
        modes = select_modes(message, version)
        assert len(list(encode_mixed(modes, version))) == mixed_encoding_length(modes, version)


def test_integer_secant_method():
    for root in range(1, 41):
        # Simple function with exactly one root. It's only supposed to work
        # for strictly increasing functions anyways.
        func = lambda n: int(100 * (n**0.5 - root**0.5))
        assert integer_secant_method(func, 1, 40) == root
