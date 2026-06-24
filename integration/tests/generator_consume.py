# Regression test for generator resumption when consumed outside a `for` loop.

def gen_simple():
    yield 1
    yield 2
    yield 3


# list()/tuple() resume from inside the constructor call (a deeper frame).
assert list(gen_simple()) == [1, 2, 3]
assert tuple(gen_simple()) == (1, 2, 3)

# Top-level next() across several resumes.
it = gen_simple()
assert next(it) == 1
assert next(it) == 2
assert next(it) == 3


# Generator with parameters and locals carried across yields: exercises a
# non-zero locals_count when the frame is rebased.
def running_total(n):
    total = 0
    for i in range(n):
        total += i
        yield total


assert list(running_total(5)) == [0, 1, 3, 6, 10]


# Nested `yield from` consumed by list().
def inner():
    yield from [1, 2, 3]


def outer():
    yield from inner()
    yield 4


assert list(outer()) == [1, 2, 3, 4]


# Two generators alive at once, advanced in interleaved order.
def tagged(tag):
    yield tag
    yield tag + 10


a = tagged(1)
b = tagged(2)
assert next(a) == 1
assert next(b) == 2
assert next(a) == 11
assert next(b) == 12


# The `for` path (which already worked) must keep working.
collected = []
for value in gen_simple():
    collected.append(value)
assert collected == [1, 2, 3]


# A generator consumed from inside another function-call frame.
def consume_first(iterator):
    return next(iterator)


assert consume_first(gen_simple()) == 1
