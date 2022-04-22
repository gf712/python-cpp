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

    def __exit__(self, type, value, tb):
        print("Exiting file", type, value, tb, sep=', ')

with Mock("test.py") as f:
    a = f.readlines()

assert a == ["it works", "EOF"], "Expected Wrapper.readlines to return the list [it works, EOF]"