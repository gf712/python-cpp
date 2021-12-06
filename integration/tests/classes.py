class A:
    a = 1
    def __new__(cls):
        assert A == cls
        return object.__new__(cls)

    def __init__(self):
        self.a = 2

    def __repr__(self):
        return "foo"

class B:
    def __init__(self, value):
        self.a = value

    def __repr__(self):
        return "foob"

    def foo(self, other):
        return other

a = A()
assert a.__repr__() == "foo"
assert a.a == 2
b = B(-1)
assert b.__repr__() == "foob"
assert b.foo(a).a == a.a
assert b.a == -1