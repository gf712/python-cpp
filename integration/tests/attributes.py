class A:
    def __init__(self):
        self.a = 1

a = A()
a.a += 1
assert a.a == 2, "Failed to store attribute after inplace addition"