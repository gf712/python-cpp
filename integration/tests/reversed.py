r = reversed([1,2,3])
assert next(r) == 3
assert next(r) == 2
assert next(r) == 1
try:
    raises_stop_iteration = False
    next(r)
except StopIteration:
    raises_stop_iteration = True
finally:
    assert raises_stop_iteration
