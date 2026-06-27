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

try:
    reversed()
except TypeError:
    raised_type_error = True
else:
    raised_type_error = False
assert raised_type_error, "Expected reversed() with no arguments to raise TypeError"
