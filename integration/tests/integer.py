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

to_bytes_test()

def from_bytes_test():
    assert int.from_bytes(b"10", "little") == 12337
    assert int.from_bytes(b"10", "big") == 12592

from_bytes_test()

def big_int_addition():
    a = 12345678901234567890
    b = 67890123456789012345
    c = a + b
    assert c == 80235802358023580235

big_int_addition()
