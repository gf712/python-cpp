import _weakref
import gc

# A class that supports weakrefs (built-ins like int/list typically don't).
class Foo:
    pass

def weakref_registers_and_counts():
    f = Foo()
    r = _weakref.ref(f)
    # Holding a weakref does not bump the strong-ref count, but the runtime
    # must record the registration so weakref_count is observable.
    assert _weakref.getweakrefcount(f) == 1
    # The weakref still resolves to the target while the target is alive.
    assert r() is f

weakref_registers_and_counts()

def weakref_resolves_through_gc_collect():
    # gc.collect() forces a full mark/sweep regardless of cadence/pause
    # state, so this scope can deterministically check that the weakref
    # keeps tracking the target across collection cycles.
    f = Foo()
    r = _weakref.ref(f)
    gc.collect()
    # `f` is still on the stack here, so it must survive the collection.
    assert r() is f
    assert _weakref.getweakrefcount(f) == 1

weakref_resolves_through_gc_collect()

def gc_collect_does_not_crash_when_disabled():
    # Disabling the GC must not prevent gc.collect() from running; this
    # mirrors CPython's gc.collect() semantics.
    gc.disable()
    try:
        assert gc.isenabled() is False
        gc.collect()
    finally:
        gc.enable()
    assert gc.isenabled() is True

gc_collect_does_not_crash_when_disabled()

def weakref_wrapper_unregisters_on_its_own_collection():
    # Regression test for B8. Whenever a weakref wrapper is collected,
    # the runtime must scrub the heap's m_weakrefs table — otherwise
    # the per-target vector accumulates dangling pointers and
    # getweakrefcount lies. Create N immediately-unreachable wrappers
    # and confirm the count is zero after gc.collect(). Pre-fix this
    # interpreter reported 100/100 survived; post-fix and on CPython
    # (which refcounts wrappers away on the spot) it reports 0/100.
    N = 100
    def churn(target):
        for _ in range(N):
            _weakref.ref(target)  # result discarded
    t = Foo()
    churn(t)
    gc.collect()
    remaining = _weakref.getweakrefcount(t)
    assert remaining == 0, \
        f"expected all wrappers collected, {remaining}/{N} survived"

weakref_wrapper_unregisters_on_its_own_collection()
