# The parser seeds a left-recursion sentinel only for the rules that can re-enter themselves
# at the same position, which after the precedence-climbing and postfix-loop rewrites is just
# two: dotted_name (`import a.b.c`) and t_primary (the target side of an assignment).
#
# Those two are the reason the sentinel and grow_lr still exist, and this file is what proves
# they still work. If a rule is dropped from is_left_recursive it will recurse without
# terminating, and if a newly left-recursive rule is added without an entry the same happens
# there - so a hang here is as much a failure as a wrong answer.

# dotted_name: dotted_name '.' NAME | NAME
import os.path

assert os.path.sep == "/"

import os.path as shortcut

assert shortcut.sep == "/"

from os.path import sep

assert sep == "/"


# t_primary: the left-recursive part of an assignment target. The parser matches the longest
# prefix that is still followed by a postfix operator, and the enclosing rule takes the last
# one, so each extra link here exercises another turn of the seed-growing loop.
nested = {"x": {"y": [1, 2]}}
nested["x"]["y"][0] = 9
assert nested["x"]["y"][0] == 9
assert nested["x"]["y"][1] == 2


class Holder:
    def __init__(self):
        self.d = {"k": [0, 0]}
        self.child = None


h = Holder()
h.d["k"][0] = 5
assert h.d["k"][0] == 5

# attribute target reached through another attribute
h.child = Holder()
h.child.d["k"][1] = 7
assert h.child.d["k"][1] == 7

# a longer chain: attribute, attribute, subscript, subscript
h.child.child = Holder()
h.child.child.d["k"][0] = 11
assert h.child.child.d["k"][0] == 11

# plain attribute targets still work alongside the chained ones
h.child.child.d = {"k": [1]}
assert h.child.child.d["k"][0] == 1

print("left_recursive_rules: ok")
