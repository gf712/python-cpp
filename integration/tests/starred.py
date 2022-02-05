def func1(a, b, c):
    return a * b - c

a = [2, 3]
assert func1(1, *a) == -1

def func2(a, b, c, d):
    return a * b - c + d

args = [2]
kwargs = {"c": 3, "d": 4}
assert func2(1, *args, **kwargs) == 3