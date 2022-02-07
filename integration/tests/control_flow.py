a = True

solution = 42 if a else 21
assert solution == 42

a = False
solution = 42 if a else 21
assert solution == 21