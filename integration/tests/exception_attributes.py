# Regression test for BaseException str()/args/__traceback__ and for per-assert
# traceback line numbers (message-less asserts must not share a merged block that
# collapses their source locations).


def deepest_lineno(exc):
    tb = exc.__traceback__
    while tb.tb_next is not None:
        tb = tb.tb_next
    return tb.tb_lineno


# --- str(exc) is the message; args is the tuple; __traceback__ is exposed ---
try:
    raise ValueError("boom")
except ValueError as e:
    assert str(e) == "boom", str(e)
    assert e.args == ("boom",), e.args
    assert isinstance(e, ValueError)
    assert e.__traceback__ is not None

try:
    raise ValueError("a", "b")
except ValueError as e:
    assert e.args == ("a", "b"), e.args

try:
    raise KeyError("k")
except KeyError as e:
    assert e.args == ("k",), e.args


# --- the traceback line is the raising statement's line ---
def raises_value_error():
    raise ValueError("here")  # EXC_RAISE_LINE


try:
    raises_value_error()
except ValueError as e:
    assert deepest_lineno(e) == 35, deepest_lineno(e)


# --- a later message-less assert reports ITS OWN line, not the first assert's
#     (regression for the merged-assertion-block traceback bug) ---
def fails_on_third_assert():
    assert True
    assert True
    assert False  # EXC_ASSERT_LINE


try:
    fails_on_third_assert()
except AssertionError as e:
    assert deepest_lineno(e) == 49, deepest_lineno(e)

print("EXCEPTION_ATTRIBUTES_OK")
