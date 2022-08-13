def foo(a):
    def bar():
        return a
    return bar

assert foo(10)() == 10
assert foo([])() == []

def foo(a, b=0):
    b += 10
    def bar(c):
        return a, b + c
    return bar

assert foo(10)(10) == (10, 20)
assert foo([])(1) == ([], 11)

def captures_outer_variable():
    b = [1,2,3]
    def bar(a):
        def foo():
            return [x + a for x in b]
        return foo

    assert bar(1)() == [2,3,4]

captures_outer_variable()

def captures_args():
    def foo(a, *b):
        def wrapper(c):
            return a, b, c
        return wrapper

    a = foo(1, 2, 3)
    result = a(4)

    assert result[0] == 1
    assert result[1] == (2, 3)
    assert result[2] == 4

captures_args()

def captures_kwargs():
    def foo(a, *b, **kwargs):
        def wrapper(c):
            return a, b, c, kwargs
        return wrapper

    a = foo(1, 2, 3, d=4)
    result = a(4)

    assert result[0] == 1
    assert result[1] == (2, 3)
    assert result[2] == 4
    assert result[3] == {"d": 4}

captures_kwargs()