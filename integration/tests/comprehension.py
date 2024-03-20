path_separators = ['/', '\\']
assert all(len(sep) == 1 for sep in path_separators)

greater_than_5 = [x for x in range(10) if x > 5]
assert greater_than_5 == [6, 7, 8, 9]

between_5_and_8 = [x for x in range(10) if x > 5 if x < 8]
assert between_5_and_8 == [6, 7]

empty = [x for x in range(10) if x > 5 if x < 8 if x < 4]
assert empty == []

def comprehension_with_capture():
    b = [1,2,3]
    def bar(a):
        def foo():
            return [x + a for x in b]
        return foo
    f = bar(1)
    assert f() == [2,3,4]

comprehension_with_capture()

def comprehension_nested():
    l = [[0,1,2,3], [4, 5], [6]]
    f = [y for x in l if len(x) > 1 for y in x if y != 0]
    assert len(f) == 5
    assert f[0] == 1
    assert f[1] == 2
    assert f[2] == 3
    assert f[3] == 4
    assert f[4] == 5

comprehension_nested()

def dict_comprehension():
    d = {x: x for x in range(2)}
    assert len(d) == 2
    assert d[0] == 0
    assert d[1] == 1

dict_comprehension()
