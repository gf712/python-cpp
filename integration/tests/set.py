a = {x for x in range(10)}

for x in range(10):
    assert x in a

for x in range(10, 20):
    assert x not in a
