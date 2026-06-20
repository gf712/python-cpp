# Regression: every builtin exception type must construct (raise X(...)) without
# crashing — Exception subclasses missing their own __new__ used to inherit
# Exception::__new__ (which asserts the exact Exception type), and
# ModuleNotFoundError dereferenced a null kwargs.
#
# NB: the per-type check lives in a helper called from the loop body rather than
# inline, to avoid the (separate) FOR_ITER iterator-register clobber bug that a
# heavy loop body triggers.

builtin_exceptions = [
    BaseException, Exception, ValueError, KeyError, IndexError, TypeError,
    NameError, AttributeError, RuntimeError, NotImplementedError, ImportError,
    ModuleNotFoundError, OSError, LookupError, MemoryError, StopIteration,
    UnboundLocalError, AssertionError,
]


def check(exc_type):
    try:
        raise exc_type("msg")
    except BaseException as e:
        assert isinstance(e, exc_type), exc_type
        assert type(e) is exc_type, (type(e), exc_type)
        assert e.args == ("msg",), (exc_type, e.args)


for exc_type in builtin_exceptions:
    check(exc_type)


# subclass relationships still hold
try:
    raise RuntimeError("r")
except Exception as e:
    assert isinstance(e, RuntimeError)
    assert isinstance(e, Exception)
    assert isinstance(e, BaseException)


# constructed with no args
try:
    raise ValueError
except ValueError as e:
    assert e.args == ()


print("EXCEPTION_TYPES_OK")
