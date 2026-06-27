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

def staticmethod_arity():
    try:
        staticmethod()
    except TypeError:
        assert True
    else:
        assert False, "Expected staticmethod() with no arguments to raise TypeError"

staticmethod_arity()

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

def classmethod_arity():
    try:
        classmethod()
    except TypeError:
        assert True
    else:
        assert False, "Expected classmethod() with no arguments to raise TypeError"

classmethod_arity()

class D:
    def test(self):
        return __class__ == D

assert D().test()

def class_closure():
    def foo(value):
        class A:
            def value(self):
                return value
            bar = value
            del value
        return A

    assert foo(10)().bar() == 10

    try:
        foo(10)().value()
        assert False
    except AttributeError:
        pass
    else:
        assert False

class_closure()

def property_accessors():
    class C:
        @property
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

    c = C()
    c.x = 42
    assert c.x == 42, "property getter/setter round-trip failed"

    try:
        C.x.getter()
    except TypeError:
        assert True
    else:
        assert False, "Expected property.getter() with no arguments to raise TypeError"

property_accessors()

def type_three_arg():
    Foo = type("Foo", (), {})
    assert Foo.__name__ == "Foo", "type() should set the class name"
    assert isinstance(Foo(), Foo), "type()-created class should be instantiable"

    try:
        type("Bad", "notatuple", {})
    except TypeError:
        assert True
    else:
        assert False, "Expected type() with non-tuple bases to raise TypeError"

type_three_arg()
