a = (1, 2)
b = (1, 2)

def tuple_comparisson(a, b):
    assert a == b, "Tuples with equal elements should be equal"

tuple_comparisson(a, b)

def tuple_concatenation(a, b):
    c = a + b
    assert c == (1, 2, 1, 2), "Adding tuples should return the result of the concatentation"

    d = () + a
    assert d is a, "Adding a tuple to the empty tuple should return the non empty tuple object"

tuple_concatenation(a, b)