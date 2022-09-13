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