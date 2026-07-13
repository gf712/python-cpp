# Regression test: print's file argument must be honoured, including when
# sys.stdout is None (previously the sys.stdout None-guard ran before the
# file= keyword was parsed, so the call silently did nothing). Also covers
# calling a Python-level flush() from print, which used to crash in
# PyBoundMethod::__call__ on the null args tuple.

import sys


class Sink:
    def __init__(self):
        self.parts = []

    def write(self, s):
        self.parts.append(s)

    def flush(self):
        pass


s = Sink()
print("hello", "world", file=s)
assert "".join(s.parts) == "hello world\n", s.parts

# an explicit file= destination must win even when sys.stdout is None,
# while a plain print() must silently do nothing
stdout = sys.stdout
sys.stdout = None
s2 = Sink()
print("x", 1, file=s2, sep="-")
r = print("swallowed")
sys.stdout = stdout
assert "".join(s2.parts) == "x-1\n", s2.parts
assert r is None

# file=None falls back to sys.stdout
print("print_file: ok", file=None)
