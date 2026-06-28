def to_bytes_test():
    assert (3425).to_bytes(2, 'little') == b'a\r'
    raise_error = False
    try:
        a = 12345678901234567890
        a.to_bytes(5, "little")
    # FIXME: should be an OverflowError
    except ValueError:
        raise_error = True
    finally:
        assert raise_error, "should raise an error when converting an int that is too large for the given bytes"

    try:
        (5).to_bytes(2)
    except TypeError:
        assert True
    else:
        assert False, "Expected to_bytes with too few arguments to raise TypeError"

to_bytes_test()

def from_bytes_test():
    assert int.from_bytes(b"10", "little") == 12337
    assert int.from_bytes(b"10", "big") == 12592

    try:
        int.from_bytes(b"10")
    except TypeError:
        assert True
    else:
        assert False, "Expected from_bytes with too few arguments to raise TypeError"

from_bytes_test()

def big_int_addition():
    a = 12345678901234567890
    b = 67890123456789012345
    c = a + b
    assert c == 80235802358023580235

big_int_addition()

def int_constructor():
    assert int() == 0, "int() should be 0"
    assert int(3.7) == 3, "int(3.7) should truncate to 3"
    assert int("10") == 10, "int('10') should be 10"
    assert int("ff", 16) == 255, "int('ff', 16) should be 255"

    try:
        int(1, 2, 3)
    except TypeError:
        assert True
    else:
        assert False, "Expected int() with too many arguments to raise TypeError"

int_constructor()
