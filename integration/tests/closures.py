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
