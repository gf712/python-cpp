def simple_eval():
    a = 10
    foo = lambda x: x**2
    assert eval("foo(a)") == 100
simple_eval()

def multiple_value_eval():
    assert eval("1, 2") == (1, 2)
multiple_value_eval()

def invalid_eval():
    try:
        eval("a = 1")
        assert False, "Should raise syntax error when passing a statement to eval"
    except SyntaxError:
        pass
    else:
        assert False, "Wrong exception"
invalid_eval()

def exec_arity():
    try:
        exec()
    except TypeError:
        assert True
    else:
        assert False, "Expected exec() with no arguments to raise TypeError"

    try:
        exec(1, 2, 3, 4)
    except TypeError:
        assert True
    else:
        assert False, "Expected exec() with too many arguments to raise TypeError"
exec_arity()
