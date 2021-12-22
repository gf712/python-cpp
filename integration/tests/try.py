reaches_finally = False
try:
    print("try")
    a = 1
except TypeError:
    a = 2
    print("TypeError")
except RuntimeError:
    a = 3
    print("RuntimeError")
finally:
    reaches_finally = True
    print("finally")

assert a == 1, "Integer assignment should not raise an exception"
assert reaches_finally, "Finally should always be reached"

# reaches_finally = False
# try:
#     print("try")
#     a = 1
#     b = "a"
#     c = a + b
# except TypeError:
#     a = 2
#     print("TypeError")
# except RuntimeError:
#     a = 3
#     print("RuntimeError")
# finally:
#     print("finally")

# assert a == 2, "Integer and string addition should result in a TypeError"
# assert reaches_finally, "Finally should always be reached"
