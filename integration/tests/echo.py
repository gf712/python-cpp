def echo():
    value = 0
    while True:
        a = yield value
        yield a

x = echo()
# prime generator
x.send(None)

msg = "hello"
response = x.send(msg)
assert msg == response
