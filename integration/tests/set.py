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