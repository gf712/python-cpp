import _imp

def imp_query_tests():
    assert _imp.is_builtin("sys") == True, "sys should be reported as a builtin module"
    assert _imp.is_builtin("definitely_not_a_module") == False, "unknown module is not builtin"
    assert _imp.is_frozen("definitely_not_a_module") == False, "unknown module is not frozen"

imp_query_tests()

def imp_arity_tests():
    for fn in [_imp.is_builtin, _imp.is_frozen, _imp.create_builtin, _imp.exec_builtin]:
        try:
            fn()
        except TypeError:
            assert True
        else:
            assert False, "Expected _imp function to raise TypeError with no arguments"

    try:
        _imp.is_frozen(123)
    except TypeError:
        assert True
    else:
        assert False, "Expected _imp.is_frozen with a non-string name to raise TypeError"

imp_arity_tests()
