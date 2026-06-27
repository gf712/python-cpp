def concat(a, b):
    return a + b

assert concat("foo", "bar") == "foobar", "String concatentation failed"

assert type("string") == str, "A literal string should be of type str"

def isalnum_tests():
    assert str.isalnum("foo123"), "'foo123' should be just alphanumeric characters"
    assert "foo123".isalnum(), "'foo123' should be just alphanumeric characters"

    # TODO: when keyword 'not' is added fix line below
    assert str.isalnum("foo123_")==False, "'foo123_' should not be just alphanumeric characters"

isalnum_tests()

def isascii_tests():
    assert str.isascii("foo"), "All characters should be ASCII"
    # TODO: when keyword 'not' is added fix line below
    assert str.isascii("😃") == False, "Emoji should not be ASCII"

isascii_tests()

def isalpha_tests():
    assert str.isalpha("foo"), "'foo' should be just alpha characters"
    assert str.isalpha("foo123") == False, "'foo123' should not be just alpha characters"

isalpha_tests()

def isdigit_tests():
    assert str.isdigit("123"), "'123' should be just digits"
    assert str.isdigit("foo123") == False, "'foo123' should not be just digits"
    assert str.isdigit("FooBar") == False, "'foo Bar' should not be just digits"

isdigit_tests()

assert str.islower("foo bar"), "'foo bar' should be just lowercase"
assert str.isupper("FOOBAR"), "'FOO BAR' should be just uppercase"


assert str.capitalize("foo") == "Foo", "Failed to capitalize 'foo' using str type"

assert str.casefold("camelCase") == "camelcase", "Failed to casefold 'camelCase' using str type"

assert str("foo") == "foo", "str type instatiation failed"

def string_find_tests():
    assert str.find("foo", "f") == 0, "Failed to find 'f' pattern in 'foo'"
    assert str.find("foo", "foo") == 0, "Failed to find 'foo' pattern in 'foo'"
    assert str.find("foo", "oo") == 1, "Failed to find 'oo' pattern in 'foo'"
    assert str.find("foo123", "3") == 5, "Failed to find '3' pattern in 'foo123'"
    # TODO: enable test when negative number (unary ops) parsing is supported
    # assert str.find("foo123", "4") == -1, "Failed to report that '4' is not a substring of 'foo123'"
    assert str.find("foo123", "23", 2) == 4, "Failed to find '123' pattern in 'foo123' substring (start)"
    assert str.find("foo123", "23", 4) == 4, "Failed to find '123' pattern in 'foo123' substring (start and end)"

    try:
        str.find("foo", 1)
    except TypeError:
        assert True
    else:
        assert False, "Expected find to raise TypeError when the pattern is not a string"

    try:
        str.find("foo")
    except TypeError:
        assert True
    else:
        assert False, "Expected find to raise TypeError when called with too few arguments"

string_find_tests()

def string_rfind_tests():
    assert str.rfind("foofoo", "foo") == 3, "Failed to rfind 'foo' in 'foofoo'"
    assert str.rfind("foo", "o") == 2, "Failed to rfind 'o' in 'foo'"
    assert str.rfind("abcabc", "bc") == 4, "Failed to rfind 'bc' in 'abcabc'"
    assert str.rfind("foofoo", "foo", 1) == 3, "Failed to rfind 'foo' in 'foofoo' substring (start)"

    try:
        str.rfind("foo", 1)
    except TypeError:
        assert True
    else:
        assert False, "Expected rfind to raise TypeError when the pattern is not a string"

    try:
        str.rfind("foo")
    except TypeError:
        assert True
    else:
        assert False, "Expected rfind to raise TypeError when called with too few arguments"

string_rfind_tests()

def string_count_tests():
    assert str.count("aaa", "aa") == 1, "Failed to find pattern 'aa' once in 'aaa'"
    assert str.count("aaaa", "aa") == 2, "Failed to find pattern 'aa' twice in 'aaaa'"
    assert str.count("aaaaaaaaaabaaaaaaba", "aa") == 8, "Failed to find pattern 'aa' eight times in 'aaaaaaaaaabaaaaaaba'"

    assert str.count("aaaaaaaaaabaaaaaaba", "aa", 8) == 4, "Failed to find pattern 'aa' four times in 'aaaaaaaaaabaaaaaaba' substring"
    assert str.count("aaaaaaaaaabaaaaaaba", "aa", 8, 9) == 0, "Failed to find pattern 'aa' no occurences in 'aaaaaaaaaabaaaaaaba' substring"
    assert str.count("aaaaaaaaaabaaaaaaba", "aa", 8, 10) == 1, "Failed to find pattern 'aa' once in 'aaaaaaaaaabaaaaaaba' substring"

    assert str.count("foo123", "4") == 0, "Failed to find no occurences of pattern '4' in 'foo123'"

    try:
        str.count("foo", 1)
    except TypeError:
        assert True
    else:
        assert False, "Expected count to raise TypeError when the pattern is not a string"

    try:
        str.count("foo")
    except TypeError:
        assert True
    else:
        assert False, "Expected count to raise TypeError when called with too few arguments"

string_count_tests()

def string_endswith_tests():
    assert str.endswith("foo123", "123"), "'foo123' should end with '123'"
    # TODO: when keyword 'not' is added fix line below
    assert str.endswith("foo123", "123", 0, 3) == False, "'foo123' substring 'foo' should not end with '123'"
    assert str.endswith("foo123", "123", 3, 6), "'foo123' substring '123' should end with '123'"

    try:
        str.endswith("foo", 1)
    except TypeError:
        assert True
    else:
        assert False, "Expected endswith to raise TypeError when the suffix is not str or tuple"

    try:
        str.endswith("foo")
    except TypeError:
        assert True
    else:
        assert False, "Expected endswith to raise TypeError when called with too few arguments"

string_endswith_tests()

def string_startswith_tests():
    assert str.startswith("foo123", "foo"), "'foo123' should start with 'foo'"
    assert str.startswith("foo123", "123", 0, 3) is False, "'foo123' substring 'foo' should start with '123'"
    assert not str.startswith("foo123", "23", 3, 6), "'foo123' substring '23' should not start with '23'"
    a = ("foo", "bar", "bap")
    result = "bazzzz".startswith(a)
    assert not result, "bazzzz does not start with foo, bar or bap"
    a = ("foo", "bar", "baz")
    assert "bazzzz".startswith(a), "bazzzz starts with foo, bar or baz"

    try:
        str.startswith("foo", 1)
    except TypeError:
        assert True
    else:
        assert False, "Expected startswith to raise TypeError when the prefix is not str or tuple"

    try:
        str.startswith("foo")
    except TypeError:
        assert True
    else:
        assert False, "Expected startswith to raise TypeError when called with too few arguments"

string_startswith_tests()

def string_join_tests():
    assert str.join(".", ["www", "python", "org"]) == "www.python.org", "Failed to create string 'www.python.org' from join"
    assert str.join("", []) == "", "Failed to create an empty string from join with empty list"

    try:
        str.join(".")
    except TypeError:
        assert True
    else:
        assert False, "Expected join to raise TypeError when called with too few arguments"

string_join_tests()

assert str.lower("AbCDeF \o/") == "abcdef \o/", "Failed to create lowercase version of 'AbCDeF \o/'"

assert str.upper("AbCDeF \o/") == "ABCDEF \O/", "Failed to create uppercase version of 'AbCDeF \o/'"

def string_rpartition_tests():
    a = "foo.bar.baz"
    a_dot_partition = a.rpartition(".")
    assert len(a_dot_partition) == 3
    assert a_dot_partition[0] == "foo.bar"
    assert a_dot_partition[1] == "."
    assert a_dot_partition[2] == "baz"

    a = "foo.bar.baz"
    a_bar_partition = a.rpartition("|")
    assert len(a_bar_partition) == 3
    assert a_bar_partition[0] == ""
    assert a_bar_partition[1] == ""
    assert a_bar_partition[2] == "foo.bar.baz"

    try:
        "foo".rpartition(1)
    except TypeError:
        assert True
    else:
        assert False, "Expected rpartition to raise TypeError when the separator is not a string"

    try:
        "foo".rpartition()
    except TypeError:
        assert True
    else:
        assert False, "Expected rpartition to raise TypeError when called with too few arguments"

string_rpartition_tests()

def test_string_truthyness_behaviour():
    if "":
        assert False, "Empty strings should be falsy"

test_string_truthyness_behaviour()

def test_string_index():
    a = "foo😃"
    c = a[0]
    assert c == "f", "Failed to index on ASCII character"
    c = a[3]
    assert c == "😃", "Failed to index on unicode character"

test_string_index()

def test_rstrip():
    a = "   spacious   "
    b = a.rstrip()
    assert b == "   spacious"

    a = "mississippi"
    b = a.rstrip("ipz")
    assert b == "mississ"

    try:
        "foo".rstrip(1)
    except TypeError:
        assert True
    else:
        assert False, "Expected rstrip to raise TypeError when chars is not a string"

test_rstrip()

def test_strip():
    assert '   spacious   '.strip() == 'spacious'
    assert 'www.example.com'.strip('cmowz.') == 'example'

    comment_string = '#....... Section 3.2.1 Issue #32 .......'
    assert comment_string.strip('.#! ') == 'Section 3.2.1 Issue #32'

    try:
        "foo".strip(1)
    except TypeError:
        assert True
    else:
        assert False, "Expected strip to raise TypeError when chars is not a string"

    try:
        "foo".strip("a", "b")
    except TypeError:
        assert True
    else:
        assert False, "Expected strip to raise TypeError when called with too many arguments"

test_strip()

def test_split():
    assert '1,2,3'.split(',') == ['1', '2', '3']
    assert '1,2,3'.split(',', 1) == ['1', '2,3']
    assert '1,2,,3,'.split(',') == ['1', '2', '', '3', '']

    assert '1 2 3'.split() == ['1', '2', '3']
    assert '1 2 3'.split(None, 1) == ['1', '2 3']
    assert '   1   2   3   '.split() == ['1', '2', '3']

    try:
        "1,2,3".split(1)
    except TypeError:
        assert True
    else:
        assert False, "Expected split to raise TypeError when the separator is not a string"

    try:
        "1,2,3".split(",", "x")
    except TypeError:
        assert True
    else:
        assert False, "Expected split to raise TypeError when maxsplit is not an integer"

test_split()

def test_literal_hex_string():
    a = '\xff'
    assert ord(a) == 255
    assert len(a) == 1

test_literal_hex_string()

def test_chr():
    a = chr(0)
    assert a == '\x00'

    try:
        chr(0x10ffff + 1)
    except ValueError:
        assert True
    else:
        assert False, "Expected chr to raise ValueError when given an invalid codepoint"

    try:
        chr(-1)
    except ValueError:
        assert True
    else:
        assert False, "Expected chr to raise ValueError when given an invalid codepoint"

    try:
        chr(1.0)
    except TypeError:
        assert True
    else:
        assert False, "Expected chr to raise TypeError when given a non-integer value"

    try:
        chr("1")
    except TypeError:
        assert True
    else:
        assert False, "Expected chr to raise TypeError when given a non-integer value"

    assert chr(128515) == "😃"


test_chr()

def test_replace():
    try:
        "".replace("")
    except TypeError:
        assert True
    else:
        assert False, "Expected str.replace to raise TypeError with only one arg"

    try:
        "".replace(1, 1)
    except TypeError:
        assert True
    else:
        assert False, "Expected str.replace to raise TypeError with not str args"

    assert "123".replace("", "") == "123"
    assert "123".replace("", "456") == "456145624563456"
    assert "123".replace("", "456", 0) == "123"
    assert "123".replace("", "456", -1) == "456145624563456"
    assert "123".replace("", "456", 1) == "456123"
    assert "1231".replace("1", "", 1) == "231"
    assert "1231".replace("1", "", 2) == "23"
    assert "1231".replace("1", "123", 2) == "12323123"

test_replace()

def test_translate():
    assert "foo".translate({ord("f"): "b"}) == "boo"

test_translate()