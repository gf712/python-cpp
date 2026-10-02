# SyntaxError carries CPython's (msg, filename, lineno, offset, text) attribute
# set, and the two-argument form is what both Python code and the parser use.
e = SyntaxError("invalid syntax", ("bad.py", 5, 9, "def foo(:"))
assert e.args == ("invalid syntax", ("bad.py", 5, 9, "def foo(:")), e.args
assert e.msg == "invalid syntax", e.msg
assert e.filename == "bad.py", e.filename
assert e.lineno == 5, e.lineno
assert e.offset == 9, e.offset
assert e.text == "def foo(:", e.text
assert str(e) == "invalid syntax (bad.py, line 5)", str(e)

# Without an info tuple the location attributes stay None and str() is the message.
bare = SyntaxError("boom")
assert bare.args == ("boom",), bare.args
assert bare.msg == "boom", bare.msg
assert bare.filename is None, bare.filename
assert bare.lineno is None, bare.lineno
assert bare.offset is None, bare.offset
assert bare.text is None, bare.text
assert str(bare) == "boom", str(bare)

# With no arguments at all every attribute, msg included, is None.
empty = SyntaxError()
assert empty.msg is None, empty.msg
assert empty.filename is None, empty.filename
assert empty.lineno is None, empty.lineno
assert empty.offset is None, empty.offset
assert empty.text is None, empty.text

# Python 3.9 reports a short info tuple as an IndexError.
try:
    SyntaxError("x", ("f.py", 1))
    assert False, "expected IndexError"
except IndexError as error:
    assert str(error) == "tuple index out of range", str(error)

try:
    SyntaxError("x", foo=1)
    assert False, "expected TypeError"
except TypeError as error:
    assert str(error) == "SyntaxError() takes no keyword arguments", str(error)

# The attributes are writable, so nothing may assume their type.
e.lineno = "not an int"
assert str(e) == "invalid syntax (bad.py)", str(e)

# compile() raises the parser's SyntaxError instead of aborting.
try:
    compile("def foo(:", "<string>", "exec")
    assert False, "expected SyntaxError"
except SyntaxError as error:
    assert error.msg == "invalid syntax", error.msg
    assert error.lineno == 1, error.lineno
    assert error.offset == 9, error.offset

# lineno is user-supplied, so out-of-range values are printed, not asserted on.
negative = SyntaxError("m", ("f.py", -1, 1, "code"))
assert str(negative) == "m (f.py, line -1)", str(negative)
huge = SyntaxError("m", ("f.py", 10**30, 1, "code"))
assert str(huge) == "m (f.py, line 1000000000000000000000000000000)", str(huge)
