# Regression test: closing a path-opened FileIO after reading must not crash.
# Previously FileIO.close() called ferror() on the underlying FILE* *after*
# closing it (which resets the pointer to NULL), segfaulting on ferror(NULL).
# This is the same path the import machinery uses to read a module's source.

# Run with cwd == integration/ (as the integration runner does).
DATA = "tests/file_io_data.txt"

# 1. read inside a `with` block -> __exit__ closes the file (the crashing path).
with open(DATA, "rb") as f:
    data = f.read()
assert data == b"line1\nline2\n", data

# 2. explicit close, and close() must be idempotent (callable more than once).
g = open(DATA, "rb")
assert g.read() == b"line1\nline2\n"
g.close()
g.close()

print("file_io: ok")
