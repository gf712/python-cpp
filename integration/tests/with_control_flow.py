"""`with` interacting with loop and exception control flow.

py.with was the only flattened-region python op not declaring
RegionBranchOpInterface, even though py.br_yield already modelled the body ->
parent edge for it. Declaring it lets MLIR's region DCE and canonicalization
reason about the body's reachability, which is exactly the machinery that decides
whether a `break`/`continue` crossing the with boundary survives. These are the
shapes that exercise those edges; `with.py` covers __enter__/__exit__ protocol
details instead.
"""


class CM:
    def __init__(self, log, name):
        self.log = log
        self.name = name

    def __enter__(self):
        self.log.append(("enter", self.name))
        return self

    def __exit__(self, *args):
        self.log.append(("exit", self.name))
        return False


# break and continue leaving a with body must still run __exit__.
log = []
for i in [1, 2, 3]:
    with CM(log, i):
        if i == 1:
            continue
        if i == 3:
            break
        log.append(("body", i))
assert log == [
    ("enter", 1),
    ("exit", 1),
    ("enter", 2),
    ("body", 2),
    ("exit", 2),
    ("enter", 3),
    ("exit", 3),
], log

# with in a while body, whose else still runs on normal exit.
out = []
n = 0
while n < 2:
    n += 1
    with CM(out, n):
        out.append(("body", n))
else:
    out.append("else")
assert out == [
    ("enter", 1),
    ("body", 1),
    ("exit", 1),
    ("enter", 2),
    ("body", 2),
    ("exit", 2),
    "else",
], out

# Nested with inside a try inside a loop: both __exit__ calls run, innermost
# first, before the handler.
deep = []
for i in [1]:
    try:
        with CM(deep, "outer"):
            with CM(deep, "inner"):
                raise ValueError("x")
    except ValueError:
        deep.append("caught")
assert deep == [
    ("enter", "outer"),
    ("enter", "inner"),
    ("exit", "inner"),
    ("exit", "outer"),
    "caught",
], deep


def with_in_a_function():
    seen = []
    for i in [1, 2]:
        with CM(seen, i):
            if i == 2:
                return seen
    return seen


assert with_in_a_function() == [
    ("enter", 1),
    ("exit", 1),
    ("enter", 2),
    ("exit", 2),
], with_in_a_function()
