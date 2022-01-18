a = {"1": 1, "2": 2}

assert a.get("1") == 1
assert a.get("2") == 2
assert a.get("3") == None
assert a.get("3", 3) == 3
