class A:
    def __repr1__(self):
        return "foo"

a = A()
print(type(A))
print(A.__repr__(A()))
print(A.__repr1__(A()))
print(a)
# print(a.__repr__())