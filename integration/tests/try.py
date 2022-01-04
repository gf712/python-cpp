reaches_finally = False
try:
    print("try")
    a = 1
except ValueError:
    a = 2
    print("RuntimeError")
except TypeError:
    a = 3
    print("TypeError")
finally:
    reaches_finally = True
    print("finally")

assert a == 1, "Integer assignment should not raise an exception"
assert reaches_finally, "Finally should always be reached"

reaches_finally = False
def get_exception():
    print("foo")
    return TypeError
try:
    print("try")
    a = 1
    b = "a"
    c = a + b
except ValueError:
    a = 2
    print("ValueError")
except get_exception():
    a = 3
    print("TypeError")
finally:
    reaches_finally = True
    print("finally")

assert a == 3, "Integer and string addition should result in a TypeError"
assert reaches_finally, "Finally should always be reached"

reaches_finally = False
def get_exception():
    print("foo")
    return TypeError
try:
    print("try")
    a = 1
    b = "a"
    c = a + b
except ValueError:
    a = 2
    print("ValueError")
except Exception:
    a = 3
    print("TypeError")
finally:
    reaches_finally = True
    print("finally")

assert a == 3, "Integer and string addition should result in an exception that is derived from Exception"
assert reaches_finally, "Finally should always be reached"
