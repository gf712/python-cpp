def foo():
    yield from range(10)

acc = 0
for x in foo():
    acc += x

assert acc == 45
