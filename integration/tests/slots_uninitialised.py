# An unset __slots__ entry must read as unset, however the slot storage was recycled.
#
# The storage lives in extra bytes past the object, and both the GC and the member
# accessor test each entry against null. Slab memory is poisoned rather than zeroed,
# so if the allocator does not clear those bytes an unset slot reads back as a
# non-null garbage pointer: the accessor returns it instead of raising AttributeError,
# and the GC dereferences it.


class C:
    __slots__ = ("a", "b", "c")


def unset_raises(obj, name):
    try:
        getattr(obj, name)
    except AttributeError:
        return True
    else:
        return False


c = C()
c.a = 1
assert c.a == 1
assert unset_raises(c, "b"), "unset slot 'b' should raise AttributeError"
assert unset_raises(c, "c"), "unset slot 'c' should raise AttributeError"

# Churn so that later instances land on slots that were freed and poisoned.
for i in range(2000):
    x = C()
    x.a = i
    x.b = i
    x.c = i

for i in range(2000):
    y = C()
    y.a = i
    assert y.a == i
    assert unset_raises(y, "b"), "recycled slot 'b' should still read as unset"
    assert unset_raises(y, "c"), "recycled slot 'c' should still read as unset"

# Slots that are set must survive a collection with their values intact. The list is a
# heap object reachable only through the slot, so the GC has to trace the slot correctly.
kept = []
for i in range(500):
    z = C()
    z.a = i
    z.b = [i, i + 1]
    kept.append(z)

i = 0
for z in kept:
    assert z.a == i
    assert z.b == [i, i + 1]
    assert unset_raises(z, "c")
    i += 1

print("slots_uninitialised: ok")
