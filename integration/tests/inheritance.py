class Base:
    def __new__(cls, value):
        return object.__new__(cls)
    def __init__(self, value):
        self.value = value
    def foo(self):
        return self.value

class Base1:
    a=1
    def foo(self):
        return self.value - 1
    def bar(self):
        return self.foo()

class Derived(Base, Base1):
    def __new__(cls, value):
        return object.__new__(cls)
    def __init__(self, value):
        Base.__init__(self, value)
    def bar(self):
        return 42


b = Base(-1)
assert b.value == -1
Base.__init__(b, 2)
assert b.value == 2

d = Derived(10)
assert d.foo() == 10
assert d.bar() == 42

assert Base1.a == 1
assert d.a == 1

assert isinstance(Derived(1), object)
assert issubclass(Derived, Base)
assert issubclass(Derived, Base1)
assert issubclass(Base, Base1) == False
assert issubclass(Derived, object)

assert Base.mro() == [Base, object]
assert Derived.mro() == [Derived, Base, Base1, object]
assert Derived.__bases__ == (Base, Base1)
