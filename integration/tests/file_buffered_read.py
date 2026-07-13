# Regression test: BufferedReader.read1(n)/read(n) must return at most n
# bytes and never None. Previously the buffered fast path returned every
# buffered byte regardless of n, or None when fewer than n bytes were
# buffered.

# Run with cwd == integration/ (as the integration runner does).
DATA = "tests/file_readline_data.txt"  # b"a\nbb\nccc"

# 1. read1 returns at most n bytes; "" at EOF.
f = open(DATA, "rb")
assert f.read1(3) == b"a\nb"
assert f.read1(0) == b""
assert f.read1(100) == b"b\nccc"
assert f.read1(4) == b""
assert f.read1() == b""
f.close()

# 2. read(n) returns exactly n bytes until the stream runs out.
g = open(DATA, "rb")
assert g.read(2) == b"a\n"
assert g.read(100) == b"bb\nccc"
assert g.read(1) == b""
g.close()

# 3. read() with no argument reads everything.
h = open(DATA, "rb")
assert h.read() == b"a\nbb\nccc"
assert h.read() == b""
h.close()

# 4. readinto fills a writable buffer and returns the byte count; read-only
# buffers (bytes) must be rejected instead of silently written to.
i = open(DATA, "rb")
ba = bytearray(b"xyz")
assert i.readinto(ba) == 3
assert ba == bytearray(b"a\nb")
try:
    i.readinto(b"xxxx")
    assert False, "readinto(bytes) must raise TypeError"
except TypeError:
    pass
i.close()

print("file_buffered_read: ok")
