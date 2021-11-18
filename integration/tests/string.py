def concat(a, b):
    return a + b

assert concat("foo", "bar") == "foobar", "String concatentation failed"

# print(str.__dict__)

assert str.isalnum("foo123"), "'foo123' should be just alphanumeric characters"
# TODO: when keyword 'not' is added fix line below
assert str.isalnum("foo123_")==False, "'foo123_' should not be just alphanumeric characters"

assert str.isascii("foo"), "All characters should be ASCII"
# TODO: when keyword 'not' is added fix line below
assert str.isascii("😃") == False, "Emoji should not be ASCII"

assert str.isalpha("foo"), "'foo' should be just alpha characters"
assert str.isalpha("foo123") == False, "'foo123' should not be just alpha characters"

assert str.isdigit("123"), "'123' should be just digits"
assert str.isdigit("foo123") == False, "'foo123' should not be just digits"

assert str.islower("foo bar"), "'foo bar' should be just lowercase"
assert str.isupper("FOOBAR"), "'FOO BAR' should be just uppercase"

assert str.isdigit("FooBar") == False, "'foo Bar' should not be just digits"

assert str.capitalize("foo") == "Foo", "Failed to capitalize 'foo' using str type"

assert str.casefold("camelCase") == "camelcase", "Failed to casefold 'camelCase' using str type"

assert str("foo") == "foo", "str type instatiation failed"

assert str.find("foo", "f") == 0, "Failed to find 'f' pattern in 'foo'"
assert str.find("foo", "foo") == 0, "Failed to find 'foo' pattern in 'foo'"
assert str.find("foo", "oo") == 1, "Failed to find 'oo' pattern in 'foo'"
assert str.find("foo123", "3") == 5, "Failed to find '3' pattern in 'foo123'"
# TODO: enable test when negative number (unary ops) parsing is supported
# assert str.find("foo123", "4") == -1, "Failed to report that '4' is not a substring of 'foo123'"
assert str.find("foo123", "23", 2) == 4, "Failed to find '123' pattern in 'foo123' substring (start)"
assert str.find("foo123", "23", 4) == 4, "Failed to find '123' pattern in 'foo123' substring (start and end)"

assert str.count("aaa", "aa") == 1, "Failed to find pattern 'aa' once in 'aaa'"
assert str.count("aaaa", "aa") == 2, "Failed to find pattern 'aa' twice in 'aaaa'"
assert str.count("aaaaaaaaaabaaaaaaba", "aa") == 8, "Failed to find pattern 'aa' eight times in 'aaaaaaaaaabaaaaaaba'"

assert str.count("aaaaaaaaaabaaaaaaba", "aa", 8) == 4, "Failed to find pattern 'aa' four times in 'aaaaaaaaaabaaaaaaba' substring"
assert str.count("aaaaaaaaaabaaaaaaba", "aa", 8, 9) == 0, "Failed to find pattern 'aa' no occurences in 'aaaaaaaaaabaaaaaaba' substring"
assert str.count("aaaaaaaaaabaaaaaaba", "aa", 8, 10) == 1, "Failed to find pattern 'aa' once in 'aaaaaaaaaabaaaaaaba' substring"

assert str.count("foo123", "4") == 0, "Failed to find no occurences of pattern '4' in 'foo123'"

assert str.endswith("foo123", "123"), "'foo123' should end with '123'"
# TODO: when keyword 'not' is added fix line below
assert str.endswith("foo123", "123", 0, 3) == False, "'foo123' substring 'foo' should not end with '123'"
assert str.endswith("foo123", "123", 3, 6), "'foo123' substring '123' should end with '123'"

assert str.join(".", ["www", "python", "org"]) == "www.python.org", "Failed to create string 'www.python.org' from join"
assert str.join("", []) == "", "Failed to create an empty string from join with empty list"

assert str.lower("AbCDeF \o/") == "abcdef \o/", "Failed to create lowercase version of 'AbCDeF \o/'"

assert str.upper("AbCDeF \o/") == "ABCDEF \O/", "Failed to create uppercase version of 'AbCDeF \o/'"
