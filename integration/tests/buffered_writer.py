# BufferedWriter must respect buffer_size like CPython (Modules/_io/bufferedio.c):
# - writes that fit in the buffer's free space are only buffered (no raw write)
# - writes that don't fit first drain the buffer to the raw stream
# - payloads larger than buffer_size bypass the buffer and go straight to raw
# - the remaining tail (<= buffer_size) is buffered again
import _io

PATH = "/tmp/pycpp_buffered_writer_test.bin"


def file_contents():
    f = _io.FileIO(PATH, "rb")
    data = f.readall()
    f.close()
    return data


# 1. a small write stays in the buffer until flush
w = _io.BufferedWriter(_io.FileIO(PATH, "wb"))
assert w.write(b"abcd") == 4
assert file_contents() == b"", file_contents()
w.flush()
assert file_contents() == b"abcd", file_contents()

# 2. overflowing the buffer drains it, and an oversized payload bypasses the buffer
w = _io.BufferedWriter(_io.FileIO(PATH, "wb"), 8)
assert w.write(b"abcd") == 4
assert file_contents() == b"", file_contents()
assert w.write(b"efghijklm") == 9
assert file_contents() == b"abcdefghijklm", file_contents()

# 3. a tail smaller than buffer_size stays buffered after the drain
w = _io.BufferedWriter(_io.FileIO(PATH, "wb"), 8)
assert w.write(b"abcdef") == 6
assert w.write(b"ghi") == 3
assert file_contents() == b"abcdef", file_contents()
w.flush()
assert file_contents() == b"abcdefghi", file_contents()

# 4. an exact fit is fully buffered
w = _io.BufferedWriter(_io.FileIO(PATH, "wb"), 8)
assert w.write(b"12345678") == 8
assert file_contents() == b"", file_contents()
w.flush()
assert file_contents() == b"12345678", file_contents()

# 5. buffer_size must be strictly positive
try:
    _io.BufferedWriter(_io.FileIO(PATH, "wb"), 0)
    assert False, "expected ValueError"
except ValueError:
    pass


# 6. raw can be any duck-typed object; a write() that over-reports the byte
#    count must raise OSError instead of wrapping the unsigned byte counters
class OverReportingWriter:
    def write(self, b):
        return 1000


w = _io.BufferedWriter(OverReportingWriter(), 8)
try:
    w.write(b"0123456789abcdef")
    assert False, "expected OSError"
except OSError:
    pass


# 7. same for a negative count
class NegativeWriter:
    def write(self, b):
        return -1


w = _io.BufferedWriter(NegativeWriter(), 8)
try:
    w.write(b"0123456789abcdef")
    assert False, "expected OSError"
except OSError:
    pass

# 8. a raw write() returning 0 makes no progress; retrying forever would hang,
#    so it must raise OSError
class ZeroWriter:
    def write(self, b):
        return 0


w = _io.BufferedWriter(ZeroWriter(), 8)
try:
    w.write(b"0123456789abcdef")
    assert False, "expected OSError"
except OSError:
    pass


# 9. a raw write() returning None means the stream accepted no data without
#    blocking; that is an OSError (BlockingIOError in CPython), not a crash
class NoneWriter:
    def write(self, b):
        return None


w = _io.BufferedWriter(NoneWriter(), 8)
try:
    w.write(b"0123456789abcdef")
    assert False, "expected OSError"
except OSError:
    pass


# 10. any other non-int return from raw write() is a TypeError, not a crash
class StringWriter:
    def write(self, b):
        return "16"


w = _io.BufferedWriter(StringWriter(), 8)
try:
    w.write(b"0123456789abcdef")
    assert False, "expected TypeError"
except TypeError:
    pass

# 11. an object created via __new__ without __init__ has no raw stream; every
#     I/O method must raise ValueError instead of dereferencing it
uninitialized = _io.BufferedWriter.__new__(_io.BufferedWriter)
for method in (
    uninitialized.isatty,
    uninitialized.flush,
    lambda: uninitialized.write(b"x"),
):
    try:
        method()
        assert False, "expected ValueError"
    except ValueError:
        pass

print("buffered_writer: ok")
