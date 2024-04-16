class Wrapper:
    def __init__(self, filename):
        self._filename = filename

    def readlines(self):
        print("readlines")
        return ["it works", "EOF"]

class Mock:
    def __init__(self, filename, expected_error, expected_args):
        self.wrapper = Wrapper(filename)
        self.expected_error = expected_error
        self.expected_args = expected_args

    def __enter__(self):
        print("Opening file")
        return self.wrapper

    def __exit__(self, type_, value, tb):
        print("Exiting file", type_, value, tb, sep=', ')
        assert type_ == self.expected_error
        # assert type(value) == self.expected_error
        # assert value.args == self.expected_args
        return True

def test_no_return():
    a = None
    with Mock("test.py", ValueError, ("can't see me!",)) as f:
        a = f.readlines()
        # raising an error here should be ignored since __exit__ returns a truthy value
        raise ValueError("can't see me!")

    return a
assert test_no_return() == ["it works", "EOF"], "Expected Wrapper.readlines to return the list [it works, EOF]"

def test_return():
    with Mock("test.py", None, None) as f:
        a = f.readlines()
        return a

assert test_return() == ["it works", "EOF"], "Expected Wrapper.readlines to return the list [it works, EOF]"

def test_return_from_try1():
    with Mock("test.py", None, None) as f:
        a = f.readlines()
        try:
            try:
                a = 1 + "foo"
                return a
            except:
                return 2
            finally:
                pass
            return a
        except:
            return 10

assert test_return_from_try1() == 2


def test_return_from_try2():
    with Mock("test.py", None, None) as f:
        a = f.readlines()
        try:
            try:
                pass
            except:
                return 2
            finally:
                pass
            return a
        except:
            return 10

assert test_return_from_try2() == ["it works", "EOF"]