"""`break`/`continue` inside a loop's `else` binds to the *enclosing* loop.

A loop's else clause is not part of its body, so Python binds loop control written
there to whatever loop encloses the whole statement. The lowering used to get this
wrong in two different ways:

  * `ForLoopOpLowering` rewrote its orelse's trailing yield without checking the
    yield's kind, so the inner loop swallowed a `break` meant for the outer one —
    silently running every outer iteration.

  * the enclosing loop's walker did claim the yield when the nested loop was a
    `while`, but emitted the branch while that `py.while` was still unlowered,
    producing a cross-region block reference the verifier rejects.

Both are now handled by deferring: a loop refuses to lower while a nested loop
still holds a break/continue in its orelse, so the nested loop is flattened into
the enclosing region first and the branch is same-region by construction. That
handshake is also why both loop patterns share one pass.
"""

# break in a nested for's else breaks the OUTER for.
log = []
for outer in [1, 2, 3]:
    log.append(outer)
    for inner in []:
        pass
    else:
        break
assert log == [1], log

# Same with a while as the inner loop.
log = []
for outer in [1, 2, 3]:
    log.append(outer)
    while False:
        pass
    else:
        break
assert log == [1], log

# continue in a nested loop's else continues the OUTER loop, skipping the rest
# of the outer body.
log = []
for outer in [1, 2, 3]:
    log.append(outer)
    while False:
        pass
    else:
        continue
    log.append("after-must-not-run")
assert log == [1, 2, 3], log

log = []
for outer in [1, 2, 3]:
    log.append(outer)
    for inner in []:
        pass
    else:
        continue
    log.append("after-must-not-run")
assert log == [1, 2, 3], log

# A while as the enclosing loop.
log = []
n = 0
while n < 3:
    n += 1
    log.append(n)
    for inner in []:
        pass
    else:
        continue
    log.append("after-must-not-run")
assert log == [1, 2, 3], log

# The inner loop's own body break still binds to the inner loop, and the inner
# else is then skipped.
log = []
for outer in [1, 2]:
    for inner in [10, 20]:
        log.append((outer, inner))
        break
    else:
        log.append("inner-else-must-not-run")
    log.append(("after", outer))
assert log == [(1, 10), ("after", 1), (2, 10), ("after", 2)], log

# Three levels: the break binds to the loop enclosing the loop whose else it is,
# i.e. the middle one, so the outermost keeps iterating.
log = []
for a in [1, 2]:
    for b in [10, 20]:
        log.append((a, b))
        for c in []:
            pass
        else:
            break
    log.append(("outer", a))
assert log == [(1, 10), ("outer", 1), (2, 10), ("outer", 2)], log

# An else that neither breaks nor continues still falls through to the exit.
log = []
for outer in [1, 2]:
    for inner in []:
        pass
    else:
        log.append(("else", outer))
    log.append(("after", outer))
assert log == [("else", 1), ("after", 1), ("else", 2), ("after", 2)], log
