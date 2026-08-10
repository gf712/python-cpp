"""A while condition whose value is defined outside the condition region.

WhileOpLowering built the loop's test and cf.cond_br at the *condition value's*
definition site. That is usually inside the condition region, but not always: CSE
merges the constant behind `while True:` with an identical constant in the
enclosing function, after which py.condition tests a value defined in the
function's entry block. Inserting there put the cf.cond_br in the middle of that
block, as a second terminator, and MLIR's region DCE then segfaulted on the block
whose last operation was no longer a terminator.

py.condition is by construction the terminator of the condition region's last
block, and the value it tests necessarily dominates it, so that is where the
branch belongs.

`b = True` before the loop is what creates the constant CSE merges with — without
it the loop's `True` is unique and the bug does not appear. Reduced from
sre_parse._parse; the same fault was the long-standing `import weakref` crash.
"""


def only_exit_is_raise(a):
    b = True
    if a:
        while True:
            raise ValueError("boom")
    return b


try:
    only_exit_is_raise(True)
    raise AssertionError("should have raised")
except ValueError as e:
    assert str(e) == "boom", str(e)
assert only_exit_is_raise(False) is True


def shared_true_constant(limit):
    flag = True
    n = 0
    while True:
        n += 1
        if n >= limit:
            break
    return (n, flag)


assert shared_true_constant(3) == (3, True), shared_true_constant(3)


def shared_false_constant(a):
    flag = False
    n = 0
    while not flag:
        n += 1
        if n >= a:
            flag = True
    return n


assert shared_false_constant(2) == 2, shared_false_constant(2)


def condition_is_a_parameter(cond, limit):
    # The condition value is a block argument rather than an op result, the other
    # branch of the insertion-point choice that used to exist.
    n = 0
    while cond:
        n += 1
        if n >= limit:
            cond = False
    return n


assert condition_is_a_parameter(True, 2) == 2, condition_is_a_parameter(True, 2)
assert condition_is_a_parameter(False, 2) == 0, condition_is_a_parameter(False, 2)


def nested_loops_sharing_true(limit):
    t = True
    outer = 0
    while True:
        outer += 1
        inner = 0
        while True:
            inner += 1
            if inner >= 2:
                break
        if outer >= limit:
            break
    return (outer, inner, t)


assert nested_loops_sharing_true(2) == (2, 2, True), nested_loops_sharing_true(2)
