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
