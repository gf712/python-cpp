# `except <Type> as <name>:` must bind <name> to the exception INSTANCE,
# not to the matched type. Regression test for the MLIRGenerator handler
# codegen (py.load_exception).

raised = ValueError("boom")
try:
    raise raised
except ValueError as e:
    assert e is raised, "e must be the raised instance"
    assert isinstance(e, ValueError)
    assert type(e) is ValueError

# the bound name must be the instance even with multiple candidate handlers
try:
    raise KeyError("k")
except ValueError as e:
    bound = ("value", e)
except KeyError as e:
    bound = ("key", e)
assert bound[0] == "key"
assert isinstance(bound[1], KeyError)
assert type(bound[1]) is KeyError

# nested handlers each bind their own instance
inner_exc = TypeError("inner")
outer_exc = IndexError("outer")
try:
    try:
        raise inner_exc
    except TypeError as e:
        assert e is inner_exc
        raise outer_exc
except IndexError as e:
    assert e is outer_exc
    assert e is not inner_exc

print("EXCEPTION_BINDING_OK")
