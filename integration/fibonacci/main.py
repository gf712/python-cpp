# test file
import fibo # test module
import sys

print(sys.argv)

print(fibo.greeting)

# test fib1
print(fibo.fib1(10))

# test fib2
print(fibo.fib2(10))

# test fib3
print(fibo.fib3(10))

print("main", dir())
print("fibo", dir(fibo))