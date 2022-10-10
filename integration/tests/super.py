class A:
    def foo(self):
        return "A"

class B(A):
    def foo(self):
        return "B"

    def bar(self):
        return self.foo()

    def baz(self):
        return 42

class C(A):
    def foo(self):
        return "C"

class D(B, C):
    def foo(self):
        return "D"

class E(D):
    def foo(self):
        return "E"

def super_two_args():
    assert super(E, E()).foo() == "D"
    assert super(E, E()).bar() == "E"
    assert super(E, E()).baz() == 42

super_two_args()

class Base:
    def __init__(self):
        self.a = 1

class A(Base):
    def __init__(self):
        super().__init__()
        assert self.a == 1
        self.a = 2
        assert self.a == 2
        super(A, self).__init__()
        assert self.a == 1

def super_no_args():
    A()

super_no_args()