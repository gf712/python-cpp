def literal_byte():
    values = b"\a\a"
    assert len(values) == 2
    values = b"\a\a\\"
    assert len(values) == 3

literal_byte()

def literal_addition():
    MAGIC_NUMBER = (3425).to_bytes(2, 'little') + b'\r\n'
    assert MAGIC_NUMBER == b'a\r\r\n'

literal_addition()

def bytes_translate():
    result = bytearray(b'read this short text').translate(None, b'aeiou')
    assert result == bytearray(b'rd ths shrt txt')

bytes_translate()

def bytearray_find():
    a = bytearray(b'hello')
    assert a.find(ord('l')) == 2, "bytearray.find should return the first matching index"
    assert a.find(ord('l'), 3) == 3, "bytearray.find should honour the start argument"

    try:
        a.find(b'l')
    except TypeError:
        assert True
    else:
        assert False, "Expected bytearray.find to raise TypeError when the pattern is not an int"

    try:
        a.find()
    except TypeError:
        assert True
    else:
        assert False, "Expected bytearray.find to raise TypeError when called with no arguments"

bytearray_find()

def bytes_decode():
    assert b'hello'.decode() == 'hello', "bytes.decode() should default to utf-8"
    assert b'hello'.decode('utf-8') == 'hello', "bytes.decode('utf-8') failed"

    try:
        b'hello'.decode(1)
    except TypeError:
        assert True
    else:
        assert False, "Expected bytes.decode to raise TypeError when encoding is not a string"

    try:
        b'hello'.decode('utf-8', 'strict', 'extra')
    except TypeError:
        assert True
    else:
        assert False, "Expected bytes.decode to raise TypeError when given too many arguments"

bytes_decode()