def add(a, b):
    return a + b

assert add(1, 2) == 3, "Number addition failed"

a = 1
b = 2
assert a < b, "Less than number comparisson failed"
assert b < a == False, "Less than number comparisson should fail"

# assert 2 > 1, "Greater than number comparisson failed"