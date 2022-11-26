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
