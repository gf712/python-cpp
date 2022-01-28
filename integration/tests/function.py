def empty_function():
    pass

assert empty_function() == None, "Empty functions should return None"


def f(a, b, *args):
    return args

assert f(10, 20) == ()
assert f(10, 20, 30) == (30,)
assert f(10, 20, 30, 40) == (30, 40)