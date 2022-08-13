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