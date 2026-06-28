import posix

def fspath_tests():
    assert posix.fspath("/tmp/foo") == "/tmp/foo", "fspath should return the string path unchanged"

    try:
        posix.fspath()
    except TypeError:
        assert True
    else:
        assert False, "Expected posix.fspath() with no arguments to raise TypeError"

    try:
        posix.fspath(123)
    except TypeError:
        assert True
    else:
        assert False, "Expected posix.fspath() with a non-path argument to raise TypeError"

fspath_tests()

def listdir_tests():
    # listdir() defaults to the current directory and must not raise.
    entries = posix.listdir()
    assert len(entries) >= 0, "listdir() should return a list"
    assert posix.listdir(".") == entries, "listdir('.') should match listdir()"

    try:
        posix.listdir(".", ".")
    except TypeError:
        assert True
    else:
        assert False, "Expected posix.listdir() with too many arguments to raise TypeError"

listdir_tests()
