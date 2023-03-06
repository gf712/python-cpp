from itertools import islice

def test_islice():
    assert [x for x in islice('ABCDEFG', 2)] == ['A', 'B']
    assert [x for x in islice('ABCDEFG', 2, 4)] == ['C', 'D']
    assert [x for x in islice('ABCDEFG', 2, None)] == ['C', 'D', 'E', 'F', 'G']
    assert [x for x in islice('ABCDEFG', 0, None, 2)] == ['A', 'C', 'E', 'G']

test_islice()
