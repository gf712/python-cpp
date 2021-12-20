a = True
b = False


assert (b and a) == False, "False and True should be False"
assert (a and b) == False, "True and False should be False"
assert a and a, "True and True should be True"
assert (b and b) == False, "False and False should be False"

assert a or b, "True or False should be True"
assert b or a, "False or True should be True"
assert a or a, "True or True should be True"
assert (b or b) == False, "False or False should be False"
