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