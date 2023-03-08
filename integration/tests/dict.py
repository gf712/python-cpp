a = {"1": 1, "2": 2}

assert a.get("1") == 1
assert a.get("2") == 2
assert a.get("3") == None
assert a.get("3", 3) == 3

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
