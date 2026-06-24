# Regression: exception chaining — __cause__ (explicit, via `raise X from Y`),
# implicit __context__ (the exception being handled when a new one is raised),
# and __suppress_context__. Also covers the exception-stack hygiene that makes
# these reliable: internally-consumed StopIterations no longer linger, so a bare
# `raise` outside a handler is a RuntimeError and __context__ isn't spuriously
# populated.


# `raise X from Y` sets __cause__ to the instance and suppresses context.
try:
    try:
        raise ValueError("inner")
    except ValueError as e:
        raise KeyError("outer") from e
except KeyError as k:
    assert isinstance(k.__cause__, ValueError), k.__cause__
    assert str(k.__cause__) == "inner", str(k.__cause__)
    assert k.__suppress_context__ is True, k.__suppress_context__
    # __context__ is still set implicitly (suppress only affects display).
    assert isinstance(k.__context__, ValueError), k.__context__


# Implicit chaining without `from`: __context__ is the handled exception.
try:
    try:
        raise ValueError("v1")
    except ValueError:
        raise KeyError("k1")
except KeyError as k:
    assert k.__cause__ is None, k.__cause__
    assert isinstance(k.__context__, ValueError), k.__context__
    assert str(k.__context__) == "v1", str(k.__context__)
    assert k.__suppress_context__ is False, k.__suppress_context__


# `raise X from None` -> cause None, still suppressed.
try:
    raise KeyError("k") from None
except KeyError as k:
    assert k.__cause__ is None, k.__cause__
    assert k.__suppress_context__ is True, k.__suppress_context__


# Plain exception raised outside any handler: no cause, no context.
try:
    raise ValueError("plain")
except ValueError as e:
    assert e.__cause__ is None, e.__cause__
    assert e.__context__ is None, e.__context__
    assert e.__suppress_context__ is False, e.__suppress_context__


# A bare `raise` with no active exception is a RuntimeError (not an abort, and
# not a stale leftover exception).
try:
    raise
except RuntimeError as e:
    assert str(e) == "No active exception to reraise", str(e)


# Iterating a generator / comprehensions while handling an exception must not
# disturb the active exception (exception-stack hygiene).
def gen():
    yield 1
    yield 2
    yield 3


try:
    raise ValueError("active")
except ValueError as e:
    assert set(gen()) == {1, 2, 3}
    assert [x for x in range(4)] == [0, 1, 2, 3]
    assert {k: k * k for k in range(3)} == {0: 0, 1: 1, 2: 4}
    assert isinstance(e, ValueError) and str(e) == "active"


# Chaining attributes are writable; setting __cause__ also suppresses context.
try:
    raise ValueError("x")
except ValueError as e:
    ctx = RuntimeError("ctx")
    e.__context__ = ctx
    assert e.__context__ is ctx
    e.__cause__ = ctx
    assert e.__cause__ is ctx
    assert e.__suppress_context__ is True
    e.__suppress_context__ = False
    assert e.__suppress_context__ is False


print("EXCEPTION_CHAINING_OK")
