# Regression test: `break`/`continue` that leaves a `try`/`except` or `with`
# block inside a loop. These previously produced a `cf.br` from inside the
# still-nested python.try/with region to a block in the enclosing loop region
# (an invalid cross-region branch), which sent MLIR's region DCE into unbounded
# recursion and crashed the compiler. os.py's removedirs()/_walk() hit this,
# so `import os` segfaulted.

# break out of an except handler (while) -- the removedirs() shape.
def break_in_except_while(items):
    out = []
    i = 0
    while i < 5:
        try:
            out.append(items[i])
        except IndexError:
            break
        i += 1
    return out

assert break_in_except_while([10, 20]) == [10, 20], break_in_except_while([10, 20])

# continue out of an except handler (for).
def continue_in_except_for(items):
    out = []
    for x in items:
        try:
            if x == 0:
                raise ValueError("zero")
            out.append(x)
        except ValueError:
            continue
    return out

assert continue_in_except_for([1, 0, 2, 0, 5]) == [1, 2, 5], continue_in_except_for([1, 0, 2, 0, 5])

# break from the try body itself (not from a handler).
def break_in_try_body():
    out = []
    i = 0
    while i < 10:
        try:
            out.append(i)
            if i == 3:
                break
        except Exception:
            out.append(-1)
        i += 1
    return out

assert break_in_try_body() == [0, 1, 2, 3], break_in_try_body()

# break/continue out of a `with` must still run __exit__.
class CM:
    def __init__(self, log):
        self._log = log
    def __enter__(self):
        self._log.append("enter")
        return self
    def __exit__(self, *a):
        self._log.append("exit")
        return False

def break_out_of_with():
    log = []
    i = 0
    while i < 3:
        with CM(log):
            if i == 1:
                break
        i += 1
    return log

assert break_out_of_with() == ["enter", "exit", "enter", "exit"], break_out_of_with()

def continue_out_of_with():
    log = []
    for i in range(3):
        with CM(log):
            if i == 1:
                continue
        log.append(("after", i))
    return log

assert continue_out_of_with() == [
    "enter", "exit", ("after", 0),
    "enter", "exit",
    "enter", "exit", ("after", 2),
], continue_out_of_with()

# nested try/except with break in the inner handler -- the _walk() shape.
def nested_try_break(values):
    out = []
    it = iter(values)
    while True:
        try:
            try:
                v = next(it)
            except StopIteration:
                break
        except RuntimeError:
            out.append("runtime")
            continue
        out.append(v)
    return out

assert nested_try_break([1, 2, 3]) == [1, 2, 3], nested_try_break([1, 2, 3])

# break/continue out of a try must still run its finally first.
def break_in_except_with_finally():
    out = []
    i = 0
    while i < 5:
        try:
            out.append(("body", i))
            raise ValueError
        except ValueError:
            out.append(("except", i))
            break
        finally:
            out.append(("finally", i))
        i += 1
    return out

assert break_in_except_with_finally() == [
    ("body", 0), ("except", 0), ("finally", 0),
], break_in_except_with_finally()

def continue_with_finally():
    out = []
    for i in range(3):
        try:
            if i == 1:
                raise ValueError
            out.append(("ok", i))
        except ValueError:
            continue
        finally:
            out.append(("fin", i))
    return out

assert continue_with_finally() == [
    ("ok", 0), ("fin", 0), ("fin", 1), ("ok", 2), ("fin", 2),
], continue_with_finally()

# both break and continue, each unwinding the same finally.
def break_and_continue_with_finally():
    out = []
    for i in range(6):
        try:
            if i == 1:
                continue
            if i == 4:
                break
            out.append(("ok", i))
        finally:
            out.append(("fin", i))
    return out

assert break_and_continue_with_finally() == [
    ("ok", 0), ("fin", 0),
    ("fin", 1),
    ("ok", 2), ("fin", 2),
    ("ok", 3), ("fin", 3),
    ("fin", 4),
], break_and_continue_with_finally()

# break from the try body (no exception raised) still runs finally.
def break_in_body_with_finally():
    out = []
    i = 0
    while i < 5:
        try:
            out.append(("body", i))
            if i == 2:
                break
        finally:
            out.append(("fin", i))
        i += 1
    return out

assert break_in_body_with_finally() == [
    ("body", 0), ("fin", 0),
    ("body", 1), ("fin", 1),
    ("body", 2), ("fin", 2),
], break_in_body_with_finally()

# break through *nested* try/finally runs every finally, innermost first.
def break_through_nested_finally():
    out = []
    i = 0
    while i < 4:
        try:
            try:
                out.append(("body", i))
                if i == 1:
                    break
            finally:
                out.append(("inner-fin", i))
        finally:
            out.append(("outer-fin", i))
        i += 1
    return out

assert break_through_nested_finally() == [
    ("body", 0), ("inner-fin", 0), ("outer-fin", 0),
    ("body", 1), ("inner-fin", 1), ("outer-fin", 1),
], break_through_nested_finally()

# break written *inside* a finally exits the loop.
def break_inside_finally():
    out = []
    for i in range(4):
        try:
            out.append(("body", i))
        finally:
            out.append(("fin", i))
            if i == 1:
                break
    return out

assert break_inside_finally() == [
    ("body", 0), ("fin", 0), ("body", 1), ("fin", 1),
], break_inside_finally()

# a break inside a finally swallows an exception that is in flight -- even one
# raised by a called function.
def boom():
    raise RuntimeError("from callee")

def break_inside_finally_swallows_exception():
    out = []
    for i in range(4):
        try:
            out.append(("body", i))
            if i == 1:
                boom()
        finally:
            out.append(("fin", i))
            if i == 1:
                break
    return out

assert break_inside_finally_swallows_exception() == [
    ("body", 0), ("fin", 0), ("body", 1), ("fin", 1),
], break_inside_finally_swallows_exception()

# a continue inside a finally overrides a break in the try body.
def finally_continue_overrides_break():
    out = []
    for i in range(4):
        try:
            out.append(("body", i))
            if i == 1:
                break
        finally:
            out.append(("fin", i))
            if i == 1:
                continue
    return out

assert finally_continue_overrides_break() == [
    ("body", 0), ("fin", 0),
    ("body", 1), ("fin", 1),
    ("body", 2), ("fin", 2),
    ("body", 3), ("fin", 3),
], finally_continue_overrides_break()

# a break inside an inner finally still runs the enclosing finally.
def break_inside_inner_finally():
    out = []
    for i in range(3):
        try:
            try:
                out.append(("inner-body", i))
            finally:
                out.append(("inner-fin", i))
                if i == 1:
                    break
        finally:
            out.append(("outer-fin", i))
    return out

assert break_inside_inner_finally() == [
    ("inner-body", 0), ("inner-fin", 0), ("outer-fin", 0),
    ("inner-body", 1), ("inner-fin", 1), ("outer-fin", 1),
], break_inside_inner_finally()

print("break_continue_in_try: ok")
