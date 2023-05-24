def simple():
    name = "Python"
    msg = f"""hello {name}{name} {name}"""
    assert msg == "hello PythonPython Python"

simple()
