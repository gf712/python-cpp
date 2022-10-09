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

def foo():
    return 1

class C:
    a = staticmethod(foo)

c = C()
assert c.a() == foo()

class A:
    def __init__(self, a):
        self._a = a

    @classmethod
    def new(cls, value):
        return cls(value)

    @property
    def a(self):
        return self._a * 2

assert A(10).a == 20
assert A.new(10).a == 20

class D:
    def test(self):
        return __class__ == D

assert D().test()