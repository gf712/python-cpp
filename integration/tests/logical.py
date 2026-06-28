a = True
b = False


assert (b and a) is False, "False and True should be false"
assert (a and b) is False, "True and False should be false"
assert a and a, "True and True should be True"
assert (b and b) is False, "False and False should be false"

assert a or b, "True or False should be True"
assert b or a, "False or True should be True"
assert a or a, "True or True should be True"
assert (b or b) is False, "False or False should be false"

assert a is not b, "True should not be False"
assert (a is b) is False, "True is False should be false"

assert bool(1) is True, "bool(1) should be True"
assert bool(0) is False, "bool(0) should be False"
assert bool([]) is False, "bool of an empty list should be False"
assert bool([1]) is True, "bool of a non-empty list should be True"

def bool_arity():
    try:
        bool()
    except TypeError:
        assert True
    else:
        assert False, "Expected bool() with no arguments to raise TypeError"

    try:
        bool(1, 2)
    except TypeError:
        assert True
    else:
        assert False, "Expected bool() with too many arguments to raise TypeError"

bool_arity()
