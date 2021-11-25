def add(a, b):
    return a + b

assert add(1, 2) == 3, "Number addition failed"

a = 1
b = 2
assert a < b, "Less than number comparisson failed"
# TODO: implement parenthesis parsing for determining operator precedence
# assert (b < a) == False, "Less than number comparisson should fail"
# TODO: implement greater than
# assert 2 > 1, "Greater than number comparisson failed"

assert type(1) == int, "A literal integer should be of type int"
assert type(1.0) == float, "A literal float should be of type float"

print(1 <= 1 == True)