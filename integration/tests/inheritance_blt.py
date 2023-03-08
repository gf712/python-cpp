class A(dict):
    def __init__(self, name):
        self.name = name

a = A("foobar")
assert a.name == "foobar"
a["foo"] = "bar"
a[1] = "baz"
assert a["foo"] == "bar"
assert a[1] == "baz"
