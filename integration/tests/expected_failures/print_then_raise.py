# Buffered sys.stdout must be flushed by the interpreter's exit callbacks
# *before* the uncaught-exception traceback is printed, so with a redirected
# stdout the script's own output comes first. run_python_tests.sh checks the
# output ordering and that the exit code is non-zero.
print("before-raise")
raise RuntimeError("boom")
