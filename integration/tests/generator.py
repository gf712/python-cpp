def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a+b
        
fibo = fibonacci()
expected = [0,1,1,2,3,5,8,13,21,34,55]
for e in expected:
    assert next(fibo) == e