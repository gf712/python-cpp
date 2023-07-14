def simple():
    name = "Python"
    msg = f"""hello {name}{name} {name}"""
    assert msg == "hello PythonPython Python"

simple()

def simple_with_conversions():
    name = "Python"
    msg = f"""hello {{ {name}{name!r} {name!s} }}"""
    assert msg == "hello { Python'Python' Python }"

simple_with_conversions()