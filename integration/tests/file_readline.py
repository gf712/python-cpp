# Regression test: TextIOWrapper.readline must return one line at a time
# (including the line ending), honour the size limit, keep lines that span
# the 8192-byte read chunks intact, and return "" at EOF. Previously a single
# call concatenated lines, dropped data, or crashed on uneven line lengths.

# Run with cwd == integration/ (as the integration runner does).
DATA = "tests/file_readline_data.txt"  # b"a\nbb\nccc" (no trailing newline)
DATA_CRLF = "tests/file_readline_data_crlf.txt"  # b"one\r\ntwo\r\n"
DATA_LONG = "tests/file_readline_data_long.txt"  # b"x" * 9000 + b"\ntail\n"
DATA_STRADDLE = "tests/file_readline_data_straddle.txt"  # b"x" * 8191 + b"\r\ntail\r\n"
DATA_CR_TAIL = "tests/file_readline_data_cr_tail.txt"  # b"a\nb\r"
DATA_CHUNK_END = "tests/file_readline_data_chunk_end.txt"  # b"x" * 8191 + b"\ntail\n"

# 1. one line per call, trailing newline included, "" at EOF (and stays "").
f = open(DATA, "r")
assert f.readline() == "a\n"
assert f.readline() == "bb\n"
assert f.readline() == "ccc"
assert f.readline() == ""
assert f.readline() == ""

# 2. size limit: at most `limit` characters, the rest stays buffered.
g = open(DATA, "r")
assert g.readline(1) == "a"
assert g.readline(0) == ""
assert g.readline(100) == "\n"
assert g.readline(2) == "bb"
assert g.readline() == "\n"
assert g.readline() == "ccc"

# 3. \r\n is consumed as a single line ending (no spurious empty line).
# TODO: with newline=None CPython translates the terminator to "\n"; the
# decoder is not implemented yet, so the raw ending is returned for now.
h = open(DATA_CRLF, "r")
assert h.readline() == "one\r\n"
assert h.readline() == "two\r\n"
assert h.readline() == ""

# 4. a line longer than the 8192-byte read chunk is returned whole.
i = open(DATA_LONG, "r")
line = i.readline()
assert len(line) == 9001, len(line)
assert i.readline() == "tail\n"
assert i.readline() == ""

# 5. readlines returns the same split.
j = open(DATA, "r")
assert j.readlines() == ["a\n", "bb\n", "ccc"]

# 6. a \r\n split across the 8192-byte chunk boundary stays one line ending
# (the first chunk ends with the \r, the \n arrives with the next chunk).
k = open(DATA_STRADDLE, "r")
line = k.readline()
assert len(line) == 8193, len(line)
assert line == "x" * 8191 + "\r\n"
assert k.readline() == "tail\r\n"
assert k.readline() == ""

# 7. a \r\n straddling the size limit: the \r fits within the limit, the \n
# stays buffered and becomes its own line on the next call (CPython clamps
# the found line to size and pushes the remainder back).
m = open(DATA_CRLF, "r")
assert m.readline(4) == "one\r"
assert m.readline() == "\n"
assert m.readline() == "two\r\n"
assert m.readline() == ""

# 8. an unresolved trailing \r must not stop an earlier complete line from
# being returned (on a blocking stream the read-ahead would stall), and a
# lone \r at EOF terminates the final line.
n = open(DATA_CR_TAIL, "r")
assert n.readline() == "a\n"
assert n.readline() == "b\r"
assert n.readline() == ""

# 9. readlines keeps reading past a delimiter that lands exactly on the chunk
# boundary and drains the buffer mid-stream (the first line fills the whole
# 8192-byte chunk).
o = open(DATA_CHUNK_END, "r")
assert o.readlines() == ["x" * 8191 + "\n", "tail\n"]
p = open(DATA_CHUNK_END, "r")
assert p.readline() == "x" * 8191 + "\n"
assert p.readlines() == ["tail\n"]

print("file_readline: ok")
