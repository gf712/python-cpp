def foo():
    global a
    a = 1

foo()

assert a == 1