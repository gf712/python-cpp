def slice_construction():
    assert str(slice(5)) == "slice(None, 5, None)", "slice(stop) failed"
    assert str(slice(1, 10)) == "slice(1, 10, None)", "slice(start, stop) failed"
    assert str(slice(1, 10, 2)) == "slice(1, 10, 2)", "slice(start, stop, step) failed"

slice_construction()

def slice_errors():
    try:
        slice()
    except TypeError:
        assert True
    else:
        assert False, "Expected slice() with no arguments to raise TypeError"

    try:
        slice(1, 2, 3, 4)
    except TypeError:
        assert True
    else:
        assert False, "Expected slice with too many arguments to raise TypeError"

slice_errors()
