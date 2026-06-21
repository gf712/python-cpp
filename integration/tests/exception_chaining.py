# Regression: explicit exception chaining via `raise X from Y` exposes
# __cause__/__context__/__suppress_context__.
#
# NB: implicit __context__ chaining (auto-set to the exception being handled) is
# deliberately NOT done yet — the frame exception stack isn't reliably popped
# (internally-consumed StopIterations linger; the same bug breaks bare `raise`
# outside a handler), so the attribute is exposed and settable but defaults to
# None rather than being populated from that unreliable state.


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


# `raise X from None` -> cause None, still suppressed.
try:
    raise KeyError("k") from None
except KeyError as k:
    assert k.__cause__ is None, k.__cause__
    assert k.__suppress_context__ is True, k.__suppress_context__


# Plain exception: all three default cleanly.
try:
    raise ValueError("plain")
except ValueError as e:
    assert e.__cause__ is None, e.__cause__
    assert e.__context__ is None, e.__context__
    assert e.__suppress_context__ is False, e.__suppress_context__


# The chaining attributes are writable; setting __cause__ also suppresses context.
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


# `from` works with a freshly-constructed cause too.
try:
    raise ValueError("v") from TypeError("t")
except ValueError as e:
    assert isinstance(e.__cause__, TypeError), e.__cause__
    assert str(e.__cause__) == "t", str(e.__cause__)


print("EXCEPTION_CHAINING_OK")
