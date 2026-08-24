"""`while ... else` lowering.

A while loop's orelse region ends in a `py.br_yield` marking normal completion.
`WhileOpLowering` used to inline that region without rewriting the yield, so it
survived with no `py.while` parent left to satisfy its HasParent trait, and every
`while/else` aborted during lowering. Nested inside a `for`, the enclosing loop's
walker claimed the yield instead and branched to the *for*'s continue target from
inside the still-unlowered while region — the "reference to block defined in
another region" verifier error that blocked `import re`.

The nested case below is the shape from CPython's `sre_compile.py:471`
(`_generate_overlap_table`): a while/else inside a for, with the `break` in the
while *body*, which is what made `import re` fail.
"""

out = []

# Normal exit runs the else.
i = 0
while i < 3:
    i += 1
else:
    out.append(("normal", i))
assert out[-1] == ("normal", 3), out

# A break in the body skips the else.
i = 0
while i < 3:
    i += 1
    break
else:
    out.append("break-must-not-run")
assert out[-1] == ("normal", 3), out

# A false condition still runs the else.
while False:
    out.append("body-must-not-run")
else:
    out.append("false-from-start")
assert out[-1] == "false-from-start", out

# continue reaches the else via normal exit.
i = 0
while i < 3:
    i += 1
    continue
else:
    out.append(("continue", i))
assert out[-1] == ("continue", 3), out


def in_a_function():
    # Same shapes inside a function body, which lowers through a separate region.
    seen = []
    n = 0
    while n < 2:
        n += 1
    else:
        seen.append(n)
    while False:
        pass
    else:
        seen.append("else")
    return seen


assert in_a_function() == [2, "else"], in_a_function()


def generate_overlap_table(prefix):
    # sre_compile._generate_overlap_table, the shape that blocked `import re`.
    table = [0] * len(prefix)
    for i in range(1, len(prefix)):
        idx = table[i - 1]
        while prefix[i] != prefix[idx]:
            if idx == 0:
                table[i] = 0
                break
            idx = table[idx - 1]
        else:
            table[i] = idx + 1
    return table


assert generate_overlap_table("aab") == [0, 1, 0], generate_overlap_table("aab")
assert generate_overlap_table("abab") == [0, 0, 1, 2], generate_overlap_table("abab")
assert generate_overlap_table("aaaa") == [0, 1, 2, 3], generate_overlap_table("aaaa")
assert generate_overlap_table("abcd") == [0, 0, 0, 0], generate_overlap_table("abcd")

# while/else nested in a for: the else runs on every iteration that exits normally.
nested = []
for k in [1, 2, 3]:
    while False:
        pass
    else:
        nested.append(k)
assert nested == [1, 2, 3], nested
