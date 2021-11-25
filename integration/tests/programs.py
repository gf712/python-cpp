def fixture1():
    a = 2
    assert a == 2, "SimpleAssignment (fixture1) failed"

def fixture2():
    a = 2 ** 10
    assert a == 1024, "SimplePowerAssignment (fixture2) failed"

def fixture3():
    a = 2 * 3 + 4 * 5 * 6
    assert a == 126, "AssignmentWithMultiplicationPrecedence (fixture3) failed"

def fixture4():
    a = 1 + 2 + 5 * 2 << 10 * 2 + 1
    assert a == 27262976, "AssignmentWithBitshift (fixture4) failed"

def fixture5():
    a = 15 + 22 - 1
    b = a
    c = a + b
    assert a == 36, "MultilineNumericAssignments_a (fixture5) failed"
    assert b == 36, "MultilineNumericAssignments_b (fixture5) failed"
    assert c == 72, "MultilineNumericAssignments_c (fixture5) failed"

def fixture6():
    a = "foo"
    b = "bar"
    c = "123"
    d = a + b + c
    assert a == "foo", "MultilineStringAssignments_a (fixture6) failed"
    assert b == "bar", "MultilineStringAssignments_b (fixture6) failed"
    assert c == "123", "MultilineStringAssignments_c (fixture6) failed"
    assert d == "foobar123", "MultilineStringAssignments_d (fixture6) failed"

def fixture7():
    def add(a, b):
        return a + b
    a = 3
    b = 10
    c = add(a, b)
    a = 5
    d = add(a, b)
    e = add(a, d)
    assert a == 5, "AddFunctionDeclarationAndCall_a  (fixture7) failed"
    assert b == 10, "AddFunctionDeclarationAndCall_b (fixture7) failed"
    assert c == 13, "AddFunctionDeclarationAndCall_c (fixture7) failed"
    assert d == 15, "AddFunctionDeclarationAndCall_d (fixture7) failed"
    assert e == 20, "AddFunctionDeclarationAndCall_e (fixture7) failed"

def fixture8():
    acc = 0
    for x in [1,2,3,4]:
        acc = acc + x
    assert acc == 10, "ForLoopWithAccumulator (fixture8) failed"

def fixture9():
    acc = 0
    for x in range(100):
        acc = acc + x
    assert acc == 4950, "ForLoopWithRange (fixture9) failed"


def fixture10():
    acc_even = 0
    acc_odd = 0
    for x in range(100):
        if x % 2 == 0:
            acc_even = acc_even + x
        else:
            acc_odd = acc_odd + x
    assert acc_even == 2450, "ForLoopAccumulateEvenAndOddNumbers_even (fixture10) failed"
    assert acc_odd == 2500, "ForLoopAccumulateEvenAndOddNumbers_odd (fixture10) failed"

fixture1()
fixture2()
fixture3()
fixture4()
fixture5()
fixture6()
fixture7()
fixture8()
fixture9()
fixture10()