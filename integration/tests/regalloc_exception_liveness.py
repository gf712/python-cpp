# Regression: exception-handler edges must be modelled in liveness.
#
# An operation inside a try body can transfer to the handler, but that edge is
# not in the explicit CFG. When liveness ignored it, a value live across the try
# body via the handler path (e.g. a FOR_ITER iterator, or a value used after the
# handler) had its register reused inside the try body and was clobbered when an
# exception actually unwound.


# A for-loop whose body raises and catches: the iterator must survive the try
# body. Previously clobbered (abort in FOR_ITER / "object is not an iterator").
seen = []
for x in [1, 2, 3]:
    try:
        raise ValueError("m")
    except ValueError:
        pass
    seen.append(x)
assert seen == [1, 2, 3], seen

# Same over range() and over a list of types, with the exception bound.
total = 0
for x in range(4):
    try:
        raise ValueError("m")
    except ValueError as e:
        assert str(e) == "m"
    total += x
assert total == 6, total

for exc in [ValueError, KeyError, RuntimeError, TypeError, NameError]:
    try:
        raise exc("msg")
    except BaseException as e:
        assert isinstance(e, exc), exc
        assert e.args == ("msg",), (exc, e.args)

# Sequential try/except in one frame must not leak the prior exception's args.
try:
    raise ValueError("hello")
except ValueError as e:
    assert e.args == ("hello",), e.args
try:
    raise ValueError("a", "b")
except ValueError as e:
    assert e.args == ("a", "b"), e.args

# A recursive call whose result must survive a following try/except (the
# original minimal miscompile repro).
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


assert fib(10) == 55
try:
    raise ValueError("e")
except ValueError as e:
    assert str(e) == "e"

# Nested try/except inside a loop.
acc = 0
for x in [1, 2, 3]:
    try:
        try:
            raise ValueError(x)
        except KeyError:
            pass
    except ValueError as e:
        acc += e.args[0]
assert acc == 6, acc

print("REGALLOC_EXCEPTION_LIVENESS_OK")
