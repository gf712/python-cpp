def simple():
    name = "Python"
    msg = f"""hello {name}{name} {name}"""
    print(msg)
    assert msg == "hello PythonPython Python"

simple()
