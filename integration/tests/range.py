def test1(n):
    result = 0
    for x in range(n):
        result += x
    return result

def test2(start, stop):
    result = 0
    for x in range(start, stop):
        result += x
    return result

def test3(start, stop, step):
    result = 0
    for x in range(start, stop, step):
        result += x
    return result

assert test1(11) == 55, "Range with stop failed"
assert test2(5, 20) == 180, "Range with start and stop failed"
assert test3(5, 20, 2) == 96, "Range with start, stop and step failed"

def test_subscript():
    assert range(10)[0] == 0
    assert range(10)[9] == 9
    assert range(10)[-1] == 9
    try:
        range(10)[10]
    except IndexError:
        assert True
    else:
        assert False, "Out of bounds range access should raise an IndexError"

    r = range(0, 20, 2)
    assert r[5] == 10

test_subscript()