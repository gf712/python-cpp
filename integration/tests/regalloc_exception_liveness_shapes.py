# Regression: the exception-handler-edge liveness fix (a value live across a try
# body via the handler path must keep its register) must hold across try/except,
# try/finally, with, nested try, and except-cascade shapes — not just the simple
# FOR_ITER + try/except case. Each shape loops (register pressure) and keeps a
# value live across a multi-block / faulting try body.


# if/else inside the try body => multi-block body; loop var + accumulator survive
def if_else_body(flag):
    acc = 0
    for x in [1, 2, 3]:
        try:
            if flag:
                raise ValueError("a")
            else:
                raise KeyError("b")
        except ValueError:
            pass
        except KeyError:
            pass
        acc += x
    return acc


assert if_else_body(True) == 6, if_else_body(True)
assert if_else_body(False) == 6, if_else_body(False)


# a loop inside the try body; an outer value survives the inner loop + raise
def loop_in_try():
    out = []
    for x in [1, 2]:
        try:
            for i in range(3):
                pass
            raise ValueError(x)
        except ValueError:
            pass
        out.append(x)
    return out


assert loop_in_try() == [1, 2], loop_in_try()


# try/except/finally: finally runs on both paths; loop var survives
def try_except_finally():
    log = []
    for x in [1, 2]:
        try:
            raise ValueError(x)
        except ValueError:
            log.append(x)
        finally:
            log.append(-x)
    return log


assert try_except_finally() == [1, -1, 2, -2], try_except_finally()


# try/finally with the exception path actually taken (finally on unwind)
def try_finally_raise():
    out = []
    for x in [1, 2]:
        try:
            try:
                raise ValueError(x)
            finally:
                out.append(-x)
        except ValueError:
            out.append(x)
    return out


assert try_finally_raise() == [-1, 1, -2, 2], try_finally_raise()


class CM:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# with-statement: the body may raise; loop var survives the cleanup region
def with_body():
    res = []
    for x in [1, 2]:
        try:
            with CM():
                raise ValueError(x)
        except ValueError:
            res.append(x)
    return res


assert with_body() == [1, 2], with_body()


# nested try: inner clause does NOT match, exception propagates to outer
def nested_propagate():
    res = []
    for x in [1, 2]:
        try:
            try:
                raise ValueError(x)
            except KeyError:
                res.append("k")
        except ValueError:
            res.append(x)
    return res


assert nested_propagate() == [1, 2], nested_propagate()


# multiple except clauses (type cascade); value survives the whole try
def except_cascade(which):
    for x in [7]:
        try:
            if which == 0:
                raise ValueError(x)
            elif which == 1:
                raise KeyError(x)
            else:
                raise TypeError(x)
        except ValueError:
            return ("v", x)
        except KeyError:
            return ("k", x)
        except TypeError:
            return ("t", x)


assert except_cascade(0) == ("v", 7), except_cascade(0)
assert except_cascade(1) == ("k", 7), except_cascade(1)
assert except_cascade(2) == ("t", 7), except_cascade(2)


# a value defined before the try and used after the handler
def value_after_handler():
    out = []
    for x in [1, 2, 3]:
        keep = x * 10
        try:
            raise ValueError(x)
        except ValueError as e:
            got = e.args[0]
        out.append(keep + got)
    return out


assert value_after_handler() == [11, 22, 33], value_after_handler()


print("REGALLOC_EXCEPTION_LIVENESS_SHAPES_OK")
