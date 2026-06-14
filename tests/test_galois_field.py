import pytest

from qrpy.galois_field import EXP, LOG, GFE


def test_EXP_inverse():
    for n in range(255):
        assert LOG[EXP[n]] == n

def test_LOG_inverse():
    for n in range(1, 256):
        assert EXP[LOG[n]] == n

def test_GFE_multiplication():
    for i in range(256):
        a = GFE.exp(i)
        for j in range(256):
            assert a * GFE.exp(j) == GFE.exp(i + j), f"{i = }, {j = } failed"

def test_GFE_order():
    alpha = GFE.exp(1)
    products = [alpha]
    for i in range(256):
        products.append(products[-1] * alpha)
    # Count unique powers (make sure the field is finite)
    assert len({gfe.value for gfe in products}) == 255

def test_identities():
    zero_element = GFE(0)
    identity_element = GFE.exp(0)
    for i in range(256):
        assert GFE(i) + zero_element == GFE(i)
        assert GFE(i) * identity_element == GFE(i)
        assert GFE(i) * zero_element == zero_element
