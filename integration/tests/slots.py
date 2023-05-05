# TODO: mangle __foo
class A:
    __slots__ = ("__foo", "baz", "bar")
    def __init__(self) -> None:
        self.baz = 1
        self.__foo = 2

a = A()
assert a.baz == 1
a.baz = 10
assert a.baz == 10
try:
    a.bar
    assert False
except AttributeError:
    pass
else:
    assert False
a.bar = -1
assert a.bar == -1
