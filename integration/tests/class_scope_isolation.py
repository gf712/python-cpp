# A class body is outlined into its own function during lowering, so it must
# never reference an SSA value from the enclosing scope. Every class body ends
# by returning the __class__ cell, which is carried as a None constant in the
# Python dialect -- structurally identical to a module-level `x = None`. Before
# py.class was marked IsolatedFromAbove, CSE merged the two whenever the
# module-level constant dominated the class body, and the outlined function
# ended up returning a value defined in its parent:
#     error: 'func.return' op using value defined outside the region
# which failed lowering and then crashed. The giveaway was that only the
# *second* class broke -- the first one's constant precedes the module-level
# one, so nothing dominates it.


class A:
    pass


a = None


class B:
    pass


assert a is None
assert A().__class__ is A
assert B().__class__ is B
assert A is not B

# A function definition between the two behaves the same way: what matters is
# the module-level None, not what kind of definition precedes it.


def sandwiched():
    return 1


b = None


class C:
    pass


assert b is None
assert sandwiched() == 1
assert C().__class__ is C

# Several None-valued names interleaved with class definitions: each class body
# must still return its own class, not whichever constant happened to dominate.

d = None


class D:
    def which(self):
        return "D"


e = None


class E:
    def which(self):
        return "E"


f = None

assert d is None and e is None and f is None
assert D().which() == "D"
assert E().which() == "E"

# Other constants that are equally shareable across regions. `True`/`0`/`""`
# never triggered the original crash, but they exercise the same merge path.

g = True
h = 0
i = ""


class F:
    value = 1


assert g is True and h == 0 and i == ""
assert F.value == 1
assert F().__class__ is F

# Class bodies that legitimately close over an outer name resolve it by name
# (load_deref/load_closure), never by SSA value, so isolation must not break
# inheritance or references to earlier module-level bindings.

base_marker = None


class Base:
    marker = "base"


class Derived(Base):
    pass


assert base_marker is None
assert Derived.marker == "base"
assert issubclass(Derived, Base)
assert Derived().__class__ is Derived

# The rest of this file pins down what IsolatedFromAbove does *not* forbid.
# The trait constrains the MLIR region (no SSA values from an enclosing region),
# not Python scoping: everything the class body needs from outside arrives
# either as an operand of py.class, evaluated in the enclosing scope before the
# body runs (decorators, bases, metaclass kwargs), or by *name* through
# $captures + load_deref/load_closure (free variables), exactly as CPython's
# separate class code object does it.


def free_variable_read():
    captured = "from-outer"

    class C:
        value = captured

    return C


assert free_variable_read().value == "from-outer"


def method_default_from_enclosing_local():
    d = 42

    class C:
        def m(self, x=d):
            return x

    return C


assert method_default_from_enclosing_local()().m() == 42


def decorator_built_from_outer_value():
    tag = "tagged"

    def deco(cls):
        cls.tag = tag
        return cls

    @deco
    class C:
        pass

    return C


assert decorator_built_from_outer_value().tag == "tagged"


def base_computed_from_enclosing_local():
    class Base:
        marker = "b"

    chosen = Base

    class Derived(chosen):
        pass

    return Derived


assert base_computed_from_enclosing_local().marker == "b"


def metaclass_from_enclosing_local():
    class Meta(type):
        pass

    m = Meta

    class C(metaclass=m):
        pass

    return C


assert type(metaclass_from_enclosing_local()).__name__ == "Meta"


def comprehension_reading_enclosing_local():
    n = 3

    class C:
        items = [i for i in range(n)]

    return C


assert comprehension_reading_enclosing_local().items == [0, 1, 2]


def classes_capturing_a_loop_variable():
    out = []
    for i in range(3):

        class C:
            idx = i

        out.append(C.idx)
    return out


assert classes_capturing_a_loop_variable() == [0, 1, 2]


GLOBAL = "glob"


class UsesGlobal:
    v = GLOBAL


assert UsesGlobal.v == "glob"


# The local names here deliberately avoid colliding with any module-level name
# in this file -- see the note below about the global-vs-captured-free-variable
# bug, which is unrelated to region isolation but would otherwise mask this case.
def two_levels_of_nesting():
    outer_word = "one"

    def inner():
        inner_word = "two"

        class C:
            joined = outer_word + inner_word

        return C

    return inner()


assert two_levels_of_nesting().joined == "onetwo"


# A class nested directly inside another class body. Each py.class is lowered
# by its own run of ClassDefinitionOpLowering, and each body keeps its own
# py.class_return until then. The outer class used to rewrite *every*
# py.class_return in its subtree to func.return -- including the inner class's
# -- so by the time the inner class was lowered its terminator was gone and it
# tripped ASSERT(return_op) at FunctionPatterns.cpp. Every class here carries
# __class__ in cellvars, which is what selects that code path.


class Outer:
    class Inner:
        b = 1


assert Outer.Inner.b == 1
assert Outer.Inner().__class__ is Outer.Inner


def nested_class_in_function():
    n = 7

    class Outer:
        a = n

        class Inner:
            b = n + 1

    return Outer


assert nested_class_in_function().a == 7
assert nested_class_in_function().Inner.b == 8


class ThreeDeep:
    class Middle:
        class Innermost:
            v = "deep"


assert ThreeDeep.Middle.Innermost.v == "deep"


# The same shape where the bodies actually use the __class__ cell, so the
# LoadClosureOp rewrite of the class_return operand runs for both classes
# rather than only being selected by the cellvars check.


class OuterSuper:
    def who(self):
        return "outer"

    class InnerSuper:
        class Base:
            def who(self):
                return "base"

        class Derived(Base):
            def who(self):
                return "derived+" + super().who()


assert OuterSuper().who() == "outer"
assert OuterSuper.InnerSuper.Derived().who() == "derived+base"


# Sibling nested classes: the outer body holds more than one py.class_return
# in its subtree, so the walk has to skip each of them independently.


class TwoChildren:
    class First:
        tag = "first"

    class Second:
        tag = "second"


assert TwoChildren.First.tag == "first"
assert TwoChildren.Second.tag == "second"

# NOTE: one more case belongs here but hits a separate, pre-existing bug that
# reproduces identically on builds from before py.class was marked
# IsolatedFromAbove, so it is not a regression from region isolation:
#   - a lambda in a class body closing over an enclosing function local
#     (`k = 7; class C: f = lambda self: k`) raises NameError: name 'k' is not
#     defined -- the free variable is not threaded through the class scope to
#     the nested lambda.
# Add it here once it is fixed.
#
# A third, also pre-existing: when a name is *both* a module-level global and a
# captured free variable read by a nested function's class body, codegen aborts
# on TODO() at MLIRGenerator.cpp:327 (the store-name visibility lookup finds the
# symbol in neither the hidden nor the visible map). Minimal repro:
#     b = None
#     def f():
#         a = "one"
#         def inner():
#             b = "two"
#             class C:
#                 joined = a + b
#             return C
#         return inner()
# That is why two_levels_of_nesting() above uses distinctive local names.

print("class_scope_isolation: ok")
