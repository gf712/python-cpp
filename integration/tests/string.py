def concat(a, b):
    return a + b

assert concat("foo", "bar") == "foobar", "String concatentation failed"

assert str.capitalize("foo") == "Foo", "Failed to capitalize 'foo' using str type"
