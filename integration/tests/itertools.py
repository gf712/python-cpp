import itertools

def test_islice():
    assert [x for x in itertools.islice('ABCDEFG', 2)] == ['A', 'B']
    assert [x for x in itertools.islice('ABCDEFG', 2, 4)] == ['C', 'D']
    assert [x for x in itertools.islice('ABCDEFG', 2, None)] == ['C', 'D', 'E', 'F', 'G']
    assert [x for x in itertools.islice('ABCDEFG', 0, None, 2)] == ['A', 'C', 'E', 'G']

test_islice()

def test_permutations():
    assert [x for x in itertools.permutations(range(3))] == [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    assert [x for x in itertools.permutations(range(3), 2)] == [(0,1), (0,2), (1,0), (1,2), (2,0), (2,1)]
    assert [x for x in itertools.permutations('ABCD', 2)] == [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'A'), ('B', 'C'), ('B', 'D'), ('C', 'A'), ('C', 'B'), ('C', 'D'), ('D', 'A'), ('D', 'B'), ('D', 'C')]

test_permutations()