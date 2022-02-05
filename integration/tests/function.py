def empty_function():
    pass

assert empty_function() == None, "Empty functions should return None"


def f(a, b, *args):
    return args

assert f(10, 20) == ()
assert f(10, 20, 30) == (30,)
assert f(10, 20, 30, 40) == (30, 40)


def f(a, *args, **kwargs):
    return kwargs

assert f(10, 20) == {}
assert f(10, 20, b=30) == {'b': 30}
assert f(10, 20, b=30, c=40) == {'b': 30, 'c': 40}

def f(a, b=10, c=20):
    return a + b + c

assert f(10) == 40
assert f(10, 20) == 50
assert f(10, 20, 30) == 60


def f(a, b, *, c):
    return a + b + c

assert f(10, 20, c=30) == 60

try:
    result = f(10, 20, 30)
    assert False, "Should not be able to call keyword only argument with positional parameter"
except TypeError:
    assert True, ""