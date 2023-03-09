setcount = 0

class MyMap(dict):
    def __setitem__(self, key, value):
        global setcount
        setcount += 1
        return super().__setitem__(key, value)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __repr__(self) -> str:
        return "mydict("+super().__repr__()+")"

class Meta(type):
    @classmethod
    def __prepare__(metacls, name, bases, **kwds):
        return MyMap()

    def __new__(cls, name, bases, classdict):
        return type.__new__(cls, name, bases, classdict)

    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

class A(metaclass=Meta):
    def __init__(self) -> None:
        pass

    def foo(self):
        pass

# MyMap.__getitem__ is called four times:
# __module__
# __qualname__
# __init__
# foo
assert setcount == 4
