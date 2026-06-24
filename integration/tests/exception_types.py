# Regression: every builtin exception type must construct (raise X(...)) without
# crashing — Exception subclasses missing their own __new__ used to inherit
# Exception::__new__ (which asserts the exact Exception type), and
# ModuleNotFoundError dereferenced a null kwargs.

builtin_exceptions = [
    BaseException, Exception, ValueError, KeyError, IndexError, TypeError,
    NameError, AttributeError, RuntimeError, NotImplementedError, ImportError,
    ModuleNotFoundError, OSError, LookupError, MemoryError, StopIteration,
    UnboundLocalError, AssertionError,
]


for exc_type in builtin_exceptions:
    try:
        raise exc_type("msg")
    except BaseException as e:
        assert isinstance(e, exc_type), exc_type
        assert type(e) is exc_type, (type(e), exc_type)
        assert e.args == ("msg",), (exc_type, e.args)


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
