# Regression: containers and exceptions must repr() their elements, so strings
# render quoted ('a') the same whether stored inline or boxed.

# str() of a container uses repr() on elements
assert str(["a", "b"]) == "['a', 'b']", str(["a", "b"])
assert str(("a",)) == "('a',)", str(("a",))
assert str(("a", "b", 3)) == "('a', 'b', 3)", str(("a", "b", 3))
assert str({"a"}) == "{'a'}", str({"a"})
assert str({"k": "v"}) == "{'k': 'v'}", str({"k": "v"})

# repr() too
assert repr("abc") == "'abc'", repr("abc")
assert repr(["a", ["b"], ("c",)]) == "['a', ['b'], ('c',)]", repr(["a", ["b"], ("c",)])
assert repr({1: "x", "y": 2}) == "{1: 'x', 'y': 2}", repr({1: "x", "y": 2})

# numbers are unchanged (repr == str)
assert str([1, 2, 3]) == "[1, 2, 3]", str([1, 2, 3])
assert str((1,)) == "(1,)", str((1,))


# exception repr quotes its args; str() stays the bare message
try:
    raise ValueError("hello")
except ValueError as e:
    assert repr(e) == "ValueError('hello')", repr(e)
    assert str(e) == "hello", str(e)
    assert e.args == ("hello",), e.args

try:
    raise ValueError("a", "b")
except ValueError as e:
    assert repr(e) == "ValueError('a', 'b')", repr(e)

try:
    raise ValueError
except ValueError as e:
    assert repr(e) == "ValueError()", repr(e)

print("REPR_QUOTING_OK")
