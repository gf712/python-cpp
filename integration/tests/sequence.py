class A:
    def __init__(self) -> None:
        self.a = [1, 2, 3]
    def __len__(self) -> int:
        return len(self.a)
    def __getitem__(self, key) -> int:
        return self.a[key]
    def __setitem__(self, key, value):
        self.a[key] = value
    def __contains__(self, key) -> bool:
        return key in self.a

def len_test():
    a = A()
    assert len(a) == 3

len_test()

def getitem_test():
    a = A()
    assert a[0] == 1

getitem_test()

def setitem_test():
    a = A()
    assert a[1] != 10
    a[1] = 10
    assert a[1] == 10

setitem_test()

def contains_test():
    a = A()
    assert 1 in a
    assert 2 in a
    assert 3 in a
    assert 4 not in a
    a.a.append(4)
    assert 4 in a

contains_test()
