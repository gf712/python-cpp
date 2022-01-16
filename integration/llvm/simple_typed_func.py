def add(a: int, b: int) -> int:
    return a + b

a = 0
for x in range(100):
    a = add(a, 1)

assert a == 100