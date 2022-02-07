a = "Hello, world!"
sentence_too_long = False
if (n := len(a)) > 15:
    sentence_too_long = True

assert sentence_too_long == False
assert n == len(a)