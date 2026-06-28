ModuleType = type(__import__("sys"))

def module_construction():
    m = ModuleType("mymod", "docs")
    assert m.__name__ == "mymod", "module name should be set"
    assert m.__doc__ == "docs", "module doc should be set"

    m2 = ModuleType("noname")
    assert m2.__name__ == "noname", "module should be constructible without a doc"

module_construction()

def module_errors():
    try:
        ModuleType()
    except TypeError:
        assert True
    else:
        assert False, "Expected module() with no arguments to raise TypeError"

    try:
        ModuleType(123)
    except TypeError:
        assert True
    else:
        assert False, "Expected module() with a non-string name to raise TypeError"

module_errors()
