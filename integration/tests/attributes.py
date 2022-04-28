class A:
    def __init__(self):
        print("here")
        self.a = 1

print("A.__init__", A.__init__(A()))
a = A()
print("a.a", a.a)
a.a += 1
assert a.a == 2, "Failed to store attribute after inplace addition"