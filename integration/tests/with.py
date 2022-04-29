class Wrapper:
    def __init__(self, filename):
        self._filename = filename

    def readlines(self):
        print("readlines")
        return ["it works", "EOF"]

class Mock:
    def __init__(self, filename):
        self.wrapper = Wrapper(filename)

    def __enter__(self):
        print("Opening file")
        return self.wrapper

    def __exit__(self, type_, value, tb):
        print("Exiting file", type_, value, tb, sep=', ')
        assert type_ == ValueError
        assert type(value) == ValueError
        assert value.args == ("can't see me!",)
        return True

with Mock("test.py") as f:
    a = f.readlines()
    # raising an error here should be ignored since __exit__ returns a truthy value
    raise ValueError("can't see me!")

print(a)
assert a == ["it works", "EOF"], "Expected Wrapper.readlines to return the list [it works, EOF]"