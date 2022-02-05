def empty_function():
    pass

assert empty_function() == None, "Empty functions should return None"


def f(a, b, *args):
    return args

assert f(10, 20) == ()
assert f(10, 20, 30) == (30,)
assert f(10, 20, 30, 40) == (30, 40)


def f1(a, *args, **kwargs):
    return kwargs

assert f1(10, 20) == {}
assert f1(10, 20, b=30) == {'b': 30}
assert f1(10, 20, b=30, c=40) == {'b': 30, 'c': 40}

def f2(a, b=10, c=20):
    return a + b + c

assert f2(10) == 40
assert f2(10, 20) == 50
assert f2(10, 20, 30) == 60


def f3(a, b, *, c):
    return a + b + c

assert f3(10, 20, c=30) == 60

try:
    result = f3(10, 20, 30)
    assert False, "Should not be able to call keyword only argument with positional parameter"
except TypeError:
    assert True, ""