import pytest

from qrpy.data_encoding import (
    to_bits,
    to_byte,
    encode_numeric,
    encode_alphanumeric,
    encode_byte,
    encode_mixed,
    terminate,
    to_data_bytestream,
)

def bitstring(bitstream) -> str:
    return "".join(str(bit) for bit in bitstream)


def test_to_bits():
    for length in range(1, 16):
        assert len(list(to_bits(0b11111111, length))) == length

def test_to_byte():
    for value in range(0x100):
        assert to_byte(tuple(to_bits(value, 8))) == value  # type: ignore

def test_encode_numeric():
    assert (
        bitstring(encode_numeric("01234567", version=1))
        #  <mode> <  length  > <   012    > <   345    > <  67   >
        == "0001" "0000001000" "0000001100" "0101011001" "1000011"
    )

def test_encode_alphanumeric():
    assert (
        bitstring(encode_alphanumeric("AC-42", version=1))
        #  <mode> < length  > <    AC     > <    -4     > <  2   >
        == "0010" "000000101" "00111001110" "11100111001" "000010"
    )

def test_encode_byte():
    assert (
        bitstring(encode_byte("åäö", version=1))
        #  <mode> < length > <   å    > <   ä    > <   ö    >
        == "0100" "00000011" "11100101" "11100100" "11110110"
    )

def test_encode_mixed():
    assert (
        bitstring(
            encode_mixed(
                [
                    ("01234567", "numeric"),
                    ("AC-42", "alphanumeric"),
                    ("åäö", "byte"),
                ],
                version=1,
            )
        )
        == "00010000001000000000110001010110011000011"
        "00100000001010011100111011100111001000010"
        "010000000011111001011110010011110110"
    )

def test_terminate():
    assert (
        bitstring(terminate(encode_numeric("01234567", version=1), 100))
        #  <                    data                 > <terminator>
        == "00010000001000000000110001010110011000011"    "0000"
    )

def test_to_data_bytestream():
    bytestream = to_data_bytestream(encode_numeric("01234567", version=1), 1, "L")
    assert (
        [next(bytestream) for i in range(8)]
        #   <                        data + terminator                        ><alignment>
        == [0b00010000, 0b00100000, 0b00001100, 0b01010110, 0b01100001, 0b10000000,
        #   <padding1>  <padding2>
            0b11101100, 0b00010001]
    )
