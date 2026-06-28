a = {"1": 1, "2": 2}

assert a.get("1") == 1
assert a.get("2") == 2
assert a.get("3") == None
assert a.get("3", 3) == 3

def dict_get_arity():
    try:
        a.get()
    except TypeError:
        assert True
    else:
        assert False, "Expected dict.get to raise TypeError when called with no arguments"

dict_get_arity()

assert a["1"] == 1
a["10"] = 10
assert a["10"] == 10

try:
    a["3"] == 3
except KeyError:
    ok = True

assert ok

def dict_from_keys():
    a = dict.fromkeys([1, 2, 3])
    assert a == {1: None, 2: None, 3: None}

    a = dict.fromkeys([1, 2, 3], "a")
    assert a == {1: "a", 2: "a", 3: "a"}

    try:
        dict.fromkeys()
    except TypeError:
        assert True
    else:
        assert False, "Expected dict.fromkeys to raise TypeError when called with no arguments"

dict_from_keys()

def dict_from_map():
    b = {"a": 42, "b": 1}
    c = {"a": 43, "c": 10}
    a = {"a": 1, **b, 'foo': 'bar', **c}
    assert a["a"] == 43
    assert a["b"] == 1
    assert a["c"] == 10
    assert a["foo"] == "bar"
    assert len(a) == 4

dict_from_map()

def dict_setdefault():
    a = {"a": 1}
    assert a.setdefault("a", 10) == 1
    assert a.setdefault("b", 10) == 10
    assert a["a"] == 1
    assert a["b"] == 10

dict_setdefault()

def dict_pop_missing_key_with_failing_repr():
    # Regression: PyDict::pop formatted KeyError via key.__repr__() and
    # unconditionally unwrapped the result, aborting if __repr__ raised.
    class Bad:
        def __hash__(self):
            return 0
        def __eq__(self, other):
            return False
        def __repr__(self):
            raise ValueError("bad repr")

    d = {}
    raised = None
    try:
        d.pop(Bad())
    except KeyError:
        raised = "KeyError"
    except ValueError:
        raised = "ValueError"
    assert raised is not None, "dict.pop on missing key with failing __repr__ must not abort"

dict_pop_missing_key_with_failing_repr()

def dict_pop_arity():
    d = {"a": 1}
    assert d.pop("a", 99) == 1, "dict.pop should return the value for an existing key"
    assert d.pop("a", 99) == 99, "dict.pop should return the default for a missing key"
    try:
        d.pop()
    except TypeError:
        assert True
    else:
        assert False, "Expected dict.pop to raise TypeError when called with no arguments"

dict_pop_arity()

def dict_update_method():
    d = {"a": 1}
    d.update({"b": 2})
    assert d["a"] == 1, "dict.update should keep existing keys"
    assert d["b"] == 2, "dict.update should add new keys"
    try:
        d.update()
    except TypeError:
        assert True
    else:
        assert False, "Expected dict.update to raise TypeError when called with no arguments"

dict_update_method()
