def add(a, b):
    return a + b

assert add(1, 2) == 3, "Number addition failed"

a = 1
b = 2
c = 10.0
assert a < b, "Less than number comparisson failed"
# TODO: implement parenthesis parsing for determining operator precedence
# assert (b < a) == False, "Less than number comparisson should fail"
# TODO: implement greater than
# assert 2 > 1, "Greater than number comparisson failed"

assert type(1) == int, "A literal integer should be of type int"
assert type(1.0) == float, "A literal float should be of type float"

assert int(1) == 1, "Create a int from a literal int"
assert int(1.0) == 1, "Create a int from a literal float"

assert -a == -1, "Failed to turn positive integer to negative integer"
assert -c == -10.0, "Failed to turn positive float to negative float"

assert (a + c) * 2 == 22, "Failed to respect operator precedence with parenthesis"

assert 1E100 == 1e100, "Failed to create a number from scientific notation"
assert 0xDEADBEEF == 3735928559, "Failed to create a number from hex"
assert 0o125 == 85, "Failed to create a number from octal"
assert 0b01110001 == 113, "Failed to create a number from binary"

assert float() == 0.0, "float() with no arguments should be 0.0"
assert float(3) == 3.0, "float(3) should be 3.0"
assert float(2.5) == 2.5, "float(2.5) should be 2.5"

def float_arity():
    try:
        float(1, 2)
    except TypeError:
        assert True
    else:
        assert False, "Expected float() with too many arguments to raise TypeError"

float_arity()

def conversion_builtin_arity():
    assert ord("a") == 97, "ord('a') should be 97"
    assert chr(97) == "a", "chr(97) should be 'a'"
    assert repr(5) == "5", "repr(5) should be '5'"
    assert abs(5) == 5, "abs(5) should be 5"

    for fn in [ord, chr, hex, repr, abs]:
        try:
            fn()
        except TypeError:
            assert True
        else:
            assert False, "Expected a 1-argument builtin to raise TypeError with no arguments"
        try:
            fn(1, 2)
        except TypeError:
            assert True
        else:
            assert False, "Expected a 1-argument builtin to raise TypeError with too many arguments"

conversion_builtin_arity()
