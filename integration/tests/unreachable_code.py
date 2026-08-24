"""Statements following a terminator in the same suite.

`break`, `continue` and `return` leave the builder's insertion point in a block
that already ends in a terminator, so MLIRGenerator used to append whatever came
next in the suite *after* that terminator:

    'python.br_yield' op must be the last operation in the parent block

The verifier rejected it, but MLIR's region DCE reached it first and segfaulted
(deleteDeadness reading a null terminator), so the diagnostic never mattered.
MLIRGenerator::codegen_statements now stops at the first statement that terminates the
block, which is also what unreachable code means.

Reduced from sre_parse._parse, which is why `import re` crashed during lowering.
Nothing here asserts on the unreachable statements themselves — they cannot run;
the point is that the module compiles and the reachable behaviour is right.
"""


def after_break(values):
    seen = []
    for v in values:
        seen.append(v)
        if v == 2:
            break
            seen.append("unreachable")
            raise ValueError("unreachable")
    return seen


assert after_break([1, 2, 3]) == [1, 2], after_break([1, 2, 3])


def after_continue(values):
    seen = []
    for v in values:
        if v == 2:
            continue
            seen.append("unreachable")
        seen.append(v)
    return seen


assert after_continue([1, 2, 3]) == [1, 3], after_continue([1, 2, 3])


def after_return(a):
    return a + 1
    b = a * 2
    raise ValueError("unreachable")


assert after_return(1) == 2, after_return(1)


def after_break_in_while(a):
    n = 0
    while True:
        n += 1
        if n >= a:
            break
            n = 999
            raise ValueError("unreachable")
    return n


assert after_break_in_while(3) == 3, after_break_in_while(3)


def after_break_in_try(values):
    seen = []
    for v in values:
        try:
            seen.append(v)
            if v == 2:
                break
                raise ValueError("unreachable")
        except ValueError:
            seen.append("caught")
    return seen


assert after_break_in_try([1, 2, 3]) == [1, 2], after_break_in_try([1, 2, 3])


def after_raise(a):
    if a:
        raise ValueError("boom")
        a = 999
    return a


try:
    after_raise(True)
    raise AssertionError("should have raised")
except ValueError as e:
    assert str(e) == "boom", str(e)
assert after_raise(False) is False
