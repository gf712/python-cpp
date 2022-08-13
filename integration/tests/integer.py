def to_bytes_test():
    assert (3425).to_bytes(2, 'little') == b'a\r'
    
to_bytes_test()

def from_bytes_test():
    assert int.from_bytes(b"10", "little") == 12337
    assert int.from_bytes(b"10", "big") == 12592
    
from_bytes_test()