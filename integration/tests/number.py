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
