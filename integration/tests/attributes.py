class A:
    def __init__(self):
        print("here")
        self.a = 1

print("A.__init__", A.__init__(A()))
a = A()
print("a.a", a.a)
a.a += 1
assert a.a == 2, "Failed to store attribute after inplace addition"

def attribute_builtins():
    class B:
        pass

    b = B()
    setattr(b, "x", 5)
    assert getattr(b, "x") == 5, "setattr/getattr round-trip failed"
    assert hasattr(b, "x"), "hasattr should find a set attribute"
    assert not hasattr(b, "missing"), "hasattr should be False for a missing attribute"
    assert getattr(b, "missing", 42) == 42, "getattr should return the default for a missing attribute"

    for builtin, arg_count in [(hasattr, 2), (setattr, 3)]:
        try:
            builtin(b)
        except TypeError:
            assert True
        else:
            assert False, "Expected attribute builtin to raise TypeError on wrong arity"

attribute_builtins()