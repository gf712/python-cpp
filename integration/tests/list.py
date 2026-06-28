a = [1, 2]
b = [1, 2]

assert a == b, "Lists with equal elements should be equal"

a = [1, 2, 3]
el = a.pop()
assert el == 3, "list.pop should return the last element"
assert a == b, "list.pop should mutate the original list"

el = a.pop(-2)
assert el == 1, "list.pop with negative index should wrap around"
assert a == [2], "list.pop should mutate the original list"

el = a.pop(0)
assert el == 2, "list.pop should pop with positive index"
assert a == [], "list.pop should mutate the original list"

try:
    exception_raised = False
    a.pop()
except IndexError:
    exception_raised = True
finally:
    assert exception_raised, "list.pop with empty list should raise an IndexError"

def list_recursive_repr():
    a = []
    a.append(a)
    assert repr(a) == "[[...]]", "recursive list repr should use [...] sentinel"
    # Calling repr again must drain the visited set; otherwise nested
    # repr() would still see `a` as visited.
    assert repr(a) == "[[...]]", "recursive list repr should be idempotent"

list_recursive_repr()

def len_arity():
    assert len([1, 2, 3]) == 3, "len of a list failed"
    try:
        len()
    except TypeError:
        assert True
    else:
        assert False, "Expected len() with no arguments to raise TypeError"
    try:
        len([], [])
    except TypeError:
        assert True
    else:
        assert False, "Expected len() with too many arguments to raise TypeError"

len_arity()

def list_class_getitem():
    assert str(list[int]) == "list[int]", "list[int] generic alias failed"
    try:
        list.__class_getitem__()
    except TypeError:
        assert True
    else:
        assert False, "Expected list.__class_getitem__ to raise TypeError with no arguments"

list_class_getitem()
