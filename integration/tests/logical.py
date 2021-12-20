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
