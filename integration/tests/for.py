def test_else_branch():
    a = []
    for el in ["a", "b", "c"]:
        for _ in a:
            pass
        else:
            a.append(el)
    return a

assert test_else_branch() == ["a", "b", "c"]