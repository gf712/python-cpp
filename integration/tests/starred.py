def func(a, b, c):
    return a * b - c

a = [2, 3]
assert func(1, *a) == -1