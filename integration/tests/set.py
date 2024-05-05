a = {x for x in range(10)}

for x in range(10):
    assert x in a

for x in range(10, 20):
    assert x not in a

for x in range(10):
    assert x in a
    a.discard(x)
    assert x not in a, "set.discard should mutate the set"

assert len(a) == 0, "discarding all elements should result in an empty set"

a = {1, 2, 3}
assert a.remove(1) is None, "set.remove should return None if the element was in the set"
assert 1 not in a, "set.remove should return the element if it was in the set"

raises_key_error = False
try:
    a.remove(1)
except KeyError:
    raises_key_error = True
finally:
    assert raises_key_error, "set.remove should raise a KeyError when given a non-existent element"

a = {1, 2, 3}
a.update([4, 5, 6, 1, 2, 3])
assert len(a) == 6, "set.update should add new elements to the original set"

def set_equals():
    a = {1, 2, 3}
    b = {1, 2, 3}
    c = {1, 4, 3}
    d = [1, 2, 3]
    assert a == b
    assert not a == c
    assert not a == d

set_equals()

def set_less():
    a = {1, 2, 3}
    b = {1, 2, 3}
    c = {1, 4, 3}
    d = {1, 2, 3, 4}
    assert a <= b
    assert not a < b

    assert not a <= c
    assert not a < c

    assert a <= d
    assert a < d

set_less()

def set_union():
    x = {"a", "b", "c"}
    y = {"f", "d", "a"}
    z = {"c", "d", "e"}
    z_list = list(z)

    expected = {'f', 'a', 'c', 'b', 'd', 'e'}
    assert x.union(y, z) == expected
    assert x.union(y, z_list) == expected

    assert x | y | z == expected
    try:
        result = x | y | z_list
    except TypeError:
        assert True
    else:
        assert False

set_union()