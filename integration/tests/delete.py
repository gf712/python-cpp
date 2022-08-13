a = 1
del a

try:
    print(a)
except:
    failed = True

assert failed

a = {"a": 1}
del a["a"]

failed = False
try:
    print(a["a"])
except KeyError:
    failed = True

assert failed