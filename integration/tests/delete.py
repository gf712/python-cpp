a = 1
del a

try:
    print(a)
except NameError:
    pass
else:
    assert False, "Deleting a module scope variable should make it inaccessible"

a = {"a": 1}
del a["a"]

try:
    print(a["a"])
except KeyError:
    pass
else:
    assert False, "Deleting a dictionary with subscript should delete the entry"

a = 1
def foo():
    global a
    del a
    print(a)

try:
    foo()
except NameError:
    pass
else:
    assert False, "Deleting a module scope variable from a function declaring the variable as global, should make it inaccessible"

a = 1
def foo():
    try:
        print(a)
        del a
    except UnboundLocalError:
        pass
    else:
        assert False, "Accessing an undefined variable introduced to the local scope with the del keyword should raise UnboundLocalError"

foo()