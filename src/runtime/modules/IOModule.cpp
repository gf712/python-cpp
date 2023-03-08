#include "Modules.hpp"
#include "runtime/MemoryError.hpp"
#include "runtime/NotImplementedError.hpp"
#include "runtime/OSError.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyBytes.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyType.hpp"
#include "runtime/StopIteration.hpp"
#include "runtime/ValueError.hpp"
#include "runtime/types/api.hpp"
#include "vm/VM.hpp"

#if defined(__GLIBCXX__) || defined(__GLIBCPP__)
#include <ext/stdio_filebuf.h>
#endif
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace py {

namespace {
	static PyType *s_io_base = nullptr;
	static PyType *s_io_raw_iobase = nullptr;
	static PyType *s_io_buffered_io_base = nullptr;
	static PyType *s_io_buffered_reader = nullptr;
	static PyType *s_io_buffered_writer = nullptr;
	static PyType *s_io_buffered_rwpair = nullptr;
	static PyType *s_io_buffered_random = nullptr;
	static PyType *s_io_textiobase = nullptr;
	static PyType *s_io_incremental_newline_decoder = nullptr;
	static PyType *s_io_bytesio = nullptr;
	static PyType *s_io_fileio = nullptr;
	static PyType *s_io_stringio = nullptr;
	static PyType *s_io_textiowrapper = nullptr;

	static PyType *s_blocking_io_error = nullptr;
	static PyType *s_unsupported_operation_type = nullptr;
}// namespace

Exception *unsupported_operation(PyTuple *args, PyDict *kwargs)
{
	ASSERT(s_unsupported_operation_type);
	auto *obj = s_unsupported_operation_type->call(args, kwargs).unwrap();
	ASSERT(obj->type()->issubclass(Exception::class_type()));
	return static_cast<Exception *>(obj);
}

class IOBase : public PyBaseObject
{
	friend class ::Heap;

  private:
	IOBase() : IOBase(s_io_base->type()) {}

  protected:
	IOBase(const PyType *type) : PyBaseObject(type->underlying_type()) {}

  private:
	PyResult<bool> is_closed() const
	{
		auto result = lookup_attribute(PyString::create("__IOBase_closed").unwrap());
		if (std::get<0>(result).is_err()) return Err(std::get<0>(result).unwrap_err());
		return Ok(std::get<1>(result) == LookupAttrResult::FOUND);
	}

	PyResult<std::monostate> check_closed() const
	{
		auto closed = lookup_attribute(PyString::create("closed").unwrap());
		if (std::get<0>(closed).is_err()) return Err(std::get<0>(closed).unwrap_err());
		if (std::get<1>(closed) == LookupAttrResult::FOUND) {
			return truthy(std::get<0>(closed).unwrap(), VirtualMachine::the().interpreter())
				.and_then([](bool closed) -> PyResult<std::monostate> {
					if (!closed) return Ok(std::monostate{});
					return Err(value_error("I/O operation on closed file."));
				});
		} else {
			return Ok(std::monostate{});
		}
	}

  public:
	static constexpr std::string_view __doc__ =
		"The abstract base class for all I/O classes, acting on streams of\n"
		"bytes. There is no public constructor.\n"
		"\n"
		"This class provides dummy implementations for many methods that\n"
		"derived classes can override selectively; the default implementations\n"
		"represent a file that cannot be read, written or seeked.\n"
		"\n"
		"Even though IOBase does not declare read, readinto, or write because\n"
		"their signatures will vary, implementations and clients should\n"
		"consider those methods part of the interface. Also, implementations\n"
		"may raise UnsupportedOperation when operations they do not support are\n"
		"called.\n"
		"\n"
		"The basic type used for binary data read from or written to a file is\n"
		"bytes. Other bytes-like objects are accepted as method arguments too.\n"
		"In some cases (such as readinto), a writable object is required. Text\n"
		"I/O classes work with str data.\n"
		"\n"
		"Note that calling any method (except additional calls to close(),\n"
		"which are ignored) on a closed stream should raise a ValueError.\n"
		"\n"
		"IOBase (and its subclasses) support the iterator protocol, meaning\n"
		"that an IOBase object can be iterated over yielding the lines in a\n"
		"stream.\n"
		"\n"
		"IOBase also supports the :keyword:`with` statement. In this example,\n"
		"fp is closed after the suite of the with statement is complete:\n"
		"\n"
		"with open('spam.txt', 'r') as fp:\n"
		"    fp.write('Spam and eggs!')\n";

	static PyResult<IOBase *> create(const PyType *type)
	{
		auto &heap = VirtualMachine::the().heap();
		auto *result = heap.allocate<IOBase>(type);
		if (!result) { return Err(memory_error(sizeof(IOBase))); }
		return Ok(result);
	}

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *, PyDict *)
	{
		return IOBase::create(type);
	}

	PyResult<PyObject *> __iter__() const
	{
		if (auto closed = check_closed(); closed.is_err()) return Err(closed.unwrap_err());
		return Ok(const_cast<IOBase *>(this));
	}

	PyResult<PyObject *> __next__()
	{
		return get_method(PyString::create("readline").unwrap()).and_then([](PyObject *readline) {
			return readline->call(PyTuple::create().unwrap(), PyDict::create().unwrap())
				.and_then([](PyObject *line) -> PyResult<PyObject *> {
					auto m = line->as_mapping();
					if (auto size = m.unwrap().len(); size.is_err()) {
						return Err(size.unwrap_err());
					} else {
						if (size.unwrap() == 0) {
							// since we don't handle the situation where __next__ returns
							// Ok(nullptr), we make an extra allocation here for StopIteration,
							// and have the same semantics as cpython
							return Err(stop_iteration());
						} else {
							return Ok(line);
						}
					}
				});
		});
	}

	PyResult<PyObject *> close() const
	{
		auto closed = is_closed();
		if (closed.is_err()) return Err(closed.unwrap_err());
		if (closed.unwrap()) return Ok(py_none());

		auto flushed =
			get_method(PyString::create("flush").unwrap()).and_then([](PyObject *method) {
				return method->call(PyTuple::create().unwrap(), PyDict::create().unwrap());
			});

		if (flushed.is_err()) return flushed;

		return Ok(py_none());
	}

	PyResult<PyObject *> fileno() const
	{
		return Err(unsupported_operation(PyTuple::create(String{ "fileno" }).unwrap(), nullptr));
	}

	PyResult<PyObject *> flush() const
	{
		auto closed = is_closed();
		if (closed.is_err()) return Err(closed.unwrap_err());
		if (closed.unwrap()) { return Err(value_error("I/O operation on closed file.")); }
		return Ok(py_none());
	}

	PyResult<PyObject *> isatty() const
	{
		return check_closed().and_then([](auto) { return Ok(py_false()); });
	}

	PyResult<PyObject *> readline(int64_t limit) const
	{
		auto peek = lookup_attribute(PyString::create("peek").unwrap());
		if (std::get<0>(peek).is_err()) return std::get<0>(peek);
		const bool peakable = std::get<1>(peek) == LookupAttrResult::FOUND;
		std::vector<std::byte> buffer;

		while (limit < 0 || static_cast<int64_t>(buffer.size()) < limit) {
			int64_t nreadahead = 1;
			if (peakable) {
				auto readahead = std::get<0>(peek).unwrap()->call(
					PyTuple::create(Number{ 1 }).unwrap(), PyDict::create().unwrap());
				if (readahead.is_err()) return readahead;
				if (!as<PyBytes>(readahead.unwrap())) {
					// FIXME: should be a OSError
					return Err(value_error("peek() should have returned a bytes object, not '{}'",
						readahead.unwrap()->type()->name()));
				}
				auto readahead_bytes = as<PyBytes>(readahead.unwrap());
				if (!readahead_bytes->value().b.empty()) {
					const auto &bytes = readahead_bytes->value().b;
					const auto upper = limit == -1
										   ? bytes.size()
										   : std::min(static_cast<int64_t>(bytes.size()), limit);
					auto it = std::find(bytes.begin(), bytes.begin() + upper, std::byte{ '\n' });
					nreadahead = std::distance(bytes.begin(), it);
				}
			}
			auto b = get_method(PyString::create("read").unwrap())
						 .and_then([nreadahead](PyObject *read) {
							 return read->call(PyTuple::create(Number{ nreadahead }).unwrap(),
								 PyDict::create().unwrap());
						 });
			if (b.is_err()) return b;
			if (!as<PyBytes>(b.unwrap())) {
				// FIXME: should be a OSError
				return Err(value_error("read() should have returned a bytes object, not '{}'",
					b.unwrap()->type()->name()));
			}
			auto *new_bytes = as<PyBytes>(b.unwrap());
			if (new_bytes->value().b.size() == 0) { break; }

			buffer.insert(buffer.end(), new_bytes->value().b.begin(), new_bytes->value().b.end());
			if (static_cast<char>(buffer.back()) == '\n') { break; }
		}

		return PyBytes::create(Bytes{ std::move(buffer) });
	}

	PyResult<PyObject *> readlines(int64_t hint) const
	{
		auto result_ = PyList::create();
		if (result_.is_err()) return result_;
		auto *result = result_.unwrap();

		auto it_ = iter();
		if (it_.is_err()) return it_;
		auto *it = it_.unwrap();

		size_t length = 0;

		while (true) {
			auto line = it->next();
			if (line.is_err()) {
				if (line.unwrap_err()->type() == stop_iteration()->type()) {
					break;
				} else {
					return line;
				}
			}

			result->elements().push_back(line.unwrap());
			if (hint > 0) {
				if (auto m = result->as_mapping(); m.is_ok()) {
					if (auto size = m.unwrap().len(); size.is_err()) {
						return Err(size.unwrap_err());
					} else {
						length += size.unwrap();
						if (static_cast<int64_t>(length) > hint) { break; }
					}
				} else {
					return Err(m.unwrap_err());
				}
			}
		}

		return Ok(result);
	}

	PyResult<PyObject *> seek() const
	{
		// FIXME
		return Err(unsupported_operation(PyTuple::create(String{ "seek" }).unwrap(), nullptr));
	}

	PyResult<PyObject *> seekable() const { return Ok(py_false()); }

	PyResult<PyObject *> tell() const
	{
		return get_method(PyString::create("seek").unwrap()).and_then([](auto *seek) {
			return seek->call(
				PyTuple::create(Number{ 0 }, Number{ 1 }).unwrap(), PyDict::create().unwrap());
		});
	}

	PyResult<PyObject *> truncate() const
	{
		return Err(unsupported_operation(PyTuple::create(String{ "truncate" }).unwrap(), nullptr));
	}

	PyResult<PyObject *> writable() const { return Ok(py_false()); }

	PyResult<PyObject *> writelines(PyObject *lines) const
	{
		if (auto closed = check_closed(); closed.is_err()) return Err(closed.unwrap_err());

		auto iter_ = lines->iter();
		if (iter_.is_err()) return iter_;
		auto *iter = iter_.unwrap();

		while (true) {
			auto line = iter->next();
			if (line.is_err()) {
				if (line.unwrap_err()->type() == stop_iteration()->type()) {
					break;
				} else {
					return line;
				}
			}

			auto write = get_method(PyString::create("write").unwrap());
			if (write.is_err()) return write;

			auto res = write.unwrap()->call(
				PyTuple::create(line.unwrap()).unwrap(), PyDict::create().unwrap());
			if (res.is_err()) return res;
		}
		return Ok(py_none());
	}

	PyResult<PyObject *> check_seekable_() { return check_seekable(this); }

	PyResult<PyObject *> check_readable_() { return check_readable(this); }

	PyResult<PyObject *> check_writable_() { return check_writable(this); }

	PyObject *dict() const { return m_attributes; }

	PyType *static_type() const override { return s_io_base; }

	PyResult<PyObject *> __enter__(PyTuple *, PyDict *)
	{
		return check_closed()
			.and_then([this](auto) -> PyResult<PyObject *> { return Ok(this); })
			.or_else([](auto *err) -> PyResult<PyObject *> { return Err(err); });
	}

	PyResult<PyObject *> __exit__(PyTuple *, PyDict *)
	{
		return get_method(PyString::create("close").unwrap()).and_then([](PyObject *close) {
			return close->call(PyTuple::create().unwrap(), PyDict::create().unwrap());
		});
	}

	static PyType *register_type(PyModule *module)
	{
		if (!s_io_base) {
			s_io_base =
				klass<IOBase>(module, "_IOBase")
					.def("close", &IOBase::close)
					.property_readonly("closed",
						[](IOBase *self) {
							auto closed = self->is_closed();
							if (closed.is_err()) { TODO(); }
							return Ok(closed.unwrap() ? py_true() : py_false());
						})
					.property_readonly("__dict__", [](IOBase *self) { return Ok(self->dict()); })
					.def("fileno", &IOBase::fileno)
					.def("flush", &IOBase::flush)
					.def("isatty", &IOBase::isatty)
					.def("readline",
						[](IOBase *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
							ASSERT(!kwargs || kwargs->map().empty());
							ASSERT(args);
							if (args->elements().empty()) {
								return self->readline(-1);
							} else if (args->elements().size() > 1) {
								return Err(value_error(
									"BaseIO.readline expected at most one argument (got {})",
									args->elements().size()));
							} else {
								return PyObject::from(args->elements()[0])
									.and_then([self](auto *limit) -> PyResult<PyObject *> {
										if (!as<PyInteger>(limit)) { return Err(type_error("")); }
										return self->readline(as<PyInteger>(limit)->as_i64());
									});
							}
						})
					.def("readlines",
						[](IOBase *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
							ASSERT(!kwargs || kwargs->map().empty());
							ASSERT(args);
							if (args->elements().empty()) {
								return self->readline(-1);
							} else if (args->elements().size() > 1) {
								return Err(value_error(
									"BaseIO.readlines expected at most one argument (got {})",
									args->elements().size()));
							} else {
								return PyObject::from(args->elements()[0])
									.and_then([self](auto *hint) -> PyResult<PyObject *> {
										if (!as<PyInteger>(hint)) { return Err(type_error("")); }
										return self->readlines(as<PyInteger>(hint)->as_i64());
									});
							}
						})
					.def("seek", &IOBase::seek)
					.def("seekable", &IOBase::seekable)
					.def("tell", &IOBase::tell)
					.def("truncate", &IOBase::truncate)
					.def("writable", &IOBase::writable)
					.def("writelines",
						[](IOBase *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
							ASSERT(!kwargs || kwargs->map().empty());
							ASSERT(args);
							if (args->elements().empty()) {
								return Err(
									value_error("BaseIO.readlines expected one argument (got 0)"));
							} else if (args->elements().size() > 1) {
								return Err(value_error(
									"BaseIO.readlines expected at most one argument (got {})",
									args->elements().size()));
							} else {
								return PyObject::from(args->elements()[0])
									.and_then([self](auto *lines) -> PyResult<PyObject *> {
										return self->writelines(lines);
									});
							}
						})
					.def("_checkSeekable", &IOBase::check_seekable_)
					.def("_checkReadable", &IOBase::check_readable_)
					.def("_checkWritable", &IOBase::check_writable_)
					.def("__enter__", &IOBase::__enter__)
					.def("__exit__", &IOBase::__exit__)
					.finalize();
		}
		module->add_symbol(PyString::create("_IOBase").unwrap(), s_io_base);
		return s_io_base;
	}

  protected:
	static PyResult<PyObject *> check_seekable(PyObject *self)
	{
		return self->get_method(PyString::create("seekable").unwrap())
			.and_then([](auto *seekable) { return seekable->call(nullptr, nullptr); })
			.and_then([](auto *result) -> PyResult<PyObject *> {
				if (result != py_true()) {
					return Err(unsupported_operation(
						PyTuple::create(String{ "File or stream is not seekable." }).unwrap(),
						nullptr));
				}
				return Ok(py_true());
			});
	}

	static PyResult<PyObject *> check_readable(PyObject *self)
	{
		return self->get_method(PyString::create("readable").unwrap())
			.and_then([](auto *readable) { return readable->call(nullptr, nullptr); })
			.and_then([](auto *result) -> PyResult<PyObject *> {
				if (result != py_true()) {
					return Err(unsupported_operation(
						PyTuple::create(String{ "File or stream is not readable." }).unwrap(),
						nullptr));
				}
				return Ok(py_true());
			});
	}

	static PyResult<PyObject *> check_writable(PyObject *self)
	{
		return self->get_method(PyString::create("writable").unwrap())
			.and_then([](auto *writable) { return writable->call(nullptr, nullptr); })
			.and_then([](auto *result) -> PyResult<PyObject *> {
				if (result != py_true()) {
					return Err(unsupported_operation(
						PyTuple::create(String{ "File or stream is not writable." }).unwrap(),
						nullptr));
				}
				return Ok(py_true());
			});
	}
};

class RawIOBase : public PyBaseObject
{
	friend class ::Heap;

  private:
	RawIOBase() : RawIOBase(s_io_raw_iobase->type()) {}

  protected:
	RawIOBase(const PyType *type) : PyBaseObject(type->underlying_type()) {}

  public:
	static constexpr std::string_view __doc__ = "Base class for raw binary I/O.";

	static PyResult<RawIOBase *> create(const PyType *type)
	{
		auto &heap = VirtualMachine::the().heap();
		auto *result = heap.allocate<RawIOBase>(type);
		if (!result) { return Err(memory_error(sizeof(RawIOBase))); }
		return Ok(result);
	}

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *, PyDict *)
	{
		return IOBase::create(type);
	}

	PyType *static_type() const override { return s_io_raw_iobase; }

	PyResult<PyObject *> read(int64_t n)
	{
		if (n < 0) {
			return get_method(PyString::create("readall").unwrap()).and_then([](PyObject *readall) {
				return readall->call(PyTuple::create().unwrap(), PyDict::create().unwrap());
			});
		}

		auto bytes_ = PyBytes::create();
		if (bytes_.is_err()) return bytes_;
		auto *bytes = bytes_.unwrap();

		return get_method(PyString::create("readinto").unwrap())
			.and_then([bytes](PyObject *readinto) {
				return readinto->call(PyTuple::create(bytes).unwrap(), PyDict::create().unwrap());
			})
			.and_then([](PyObject *res) -> PyResult<int64_t> {
				if (!as<PyInteger>(res)) {
					return Err(type_error(
						"expected readinto to return an int, got '{}'", res->type()->name()));
				}
				return Ok(as<PyInteger>(res)->as_i64());
			})
			.and_then([bytes](const int64_t &n) {
				const auto &b = bytes->value().b;
				std::vector<std::byte> result{ b.begin(), b.begin() + n };
				return PyBytes::create(Bytes{ std::move(result) });
			});
	}

	PyResult<PyObject *> readinto() const
	{
		return Err(not_implemented_error("_RawIOBase.readinto"));
	}

	PyResult<PyObject *> write() const { return Err(not_implemented_error("_RawIOBase.write")); }

	static PyType *register_type(PyModule *module)
	{
		if (!s_io_raw_iobase) {
			s_io_raw_iobase =
				klass<RawIOBase>(module, "_RawIOBase", s_io_base)
					.def("read",
						[](RawIOBase *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
							ASSERT(!kwargs || kwargs->map().empty());
							int64_t n = -1;
							if (args && args->elements().size() > 1) {
								return Err(value_error(
									"_RawIOBase.read expected at most one argument (got {})",
									args->elements().size()));
							} else if (args && args->elements().size() == 1) {
								auto arg0 = PyObject::from(args->elements()[0]);
								if (arg0.is_err()) return arg0;
								if (!as<PyInteger>(arg0.unwrap()) && arg0.unwrap() != py_none()) {
									return Err(
										type_error("argument should be integer or None, not '{}'",
											arg0.unwrap()->type()->name()));
								}
								if (arg0.unwrap() != py_none()) {
									n = as<PyInteger>(arg0.unwrap())->as_i64();
								}
							}
							return self->read(n);
						})
					//   .def("readall", &RawIOBase::readall)
					.def("readinto", &RawIOBase::readinto)
					.def("write", &RawIOBase::write)
					.finalize();
		}
		module->add_symbol(PyString::create("_RawIOBase").unwrap(), s_io_raw_iobase);
		return s_io_raw_iobase;
	}
};

class BufferedIOBase : public IOBase
{
	friend class ::Heap;

	BufferedIOBase() : BufferedIOBase(s_io_buffered_io_base) {}

  protected:
	BufferedIOBase(PyType *type) : IOBase(type) {}

  public:
	static constexpr std::string_view __doc__ =
		"Base class for buffered IO objects.\n"
		"\n"
		"The main difference with RawIOBase is that the read() method\n"
		"supports omitting the size argument, and does not have a default\n"
		"implementation that defers to readinto().\n"
		"\n"
		"In addition, read(), readinto() and write() may raise\n"
		"BlockingIOError if the underlying raw stream is in non-blocking\n"
		"mode and not ready; unlike their raw counterparts, they will never\n"
		"return None.\n"
		"\n"
		"A typical implementation should not inherit from a RawIOBase\n"
		"implementation, but wrap one.\n";

	static PyResult<BufferedIOBase *> create(const PyType *type)
	{
		auto &heap = VirtualMachine::the().heap();
		auto *result = heap.allocate<BufferedIOBase>(const_cast<PyType *>(type));
		if (!result) { return Err(memory_error(sizeof(BufferedIOBase))); }
		return Ok(result);
	}

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *, PyDict *)
	{
		return BufferedIOBase::create(type);
	}

	PyResult<PyObject *> detach() const
	{
		return Err(unsupported_operation(PyTuple::create(String{ "detach" }).unwrap(), nullptr));
	}

	PyResult<PyObject *> read(PyTuple *, PyDict *) const
	{
		return Err(unsupported_operation(PyTuple::create(String{ "read" }).unwrap(), nullptr));
	}

	PyResult<PyObject *> read1(PyTuple *, PyDict *) const
	{
		return Err(unsupported_operation(PyTuple::create(String{ "read1" }).unwrap(), nullptr));
	}

	PyResult<PyObject *> readinto_generic(PyBuffer &buffer, bool readinto1) const
	{
		const auto method_name = readinto1 ? "read1" : "read";
		auto data =
			get_method(PyString::create(method_name).unwrap()).and_then([&buffer](PyObject *read) {
				return read->call(
					PyTuple::create(Number{ buffer.len }).unwrap(), PyDict::create().unwrap());
			});
		if (data.is_err()) return data;

		if (!as<PyBytes>(data.unwrap())) { return Err(type_error("read() should return bytes")); }
		auto data_bytes = as<PyBytes>(data.unwrap())->value().b;

		const auto len = data_bytes.size();

		if (static_cast<int64_t>(len) > buffer.len) {
			return Err(
				value_error("read() returned too much data: "
							"{} bytes requested, {} returned",
					buffer.len,
					len));
		}

		buffer.buf.b.insert(buffer.buf.b.end(), data_bytes.begin(), data_bytes.end());

		return data;
	}

	PyResult<PyObject *> readinto(PyBuffer &buffer) const
	{
		return readinto_generic(buffer, false);
	}

	PyResult<PyObject *> readinto1(PyBuffer &buffer) const
	{
		return readinto_generic(buffer, true);
	}

	PyResult<PyObject *> write(PyTuple *, PyDict *) const
	{
		return Err(unsupported_operation(PyTuple::create(String{ "write" }).unwrap(), nullptr));
	}

	PyType *static_type() const override { return s_io_buffered_io_base; }

	static PyType *register_type(PyModule *module)
	{
		if (!s_io_buffered_io_base) {
			s_io_buffered_io_base =
				klass<BufferedIOBase>(module, "_BufferedIOBase", s_io_base)
					.def("detach", &BufferedIOBase::detach)
					.def("read", &BufferedIOBase::read)
					.def("read1", &BufferedIOBase::read1)
					.def("readinto",
						[](BufferedIOBase *self,
							PyTuple *args,
							PyDict *kwargs) -> py::PyResult<py::PyObject *> {
							ASSERT(!kwargs || kwargs->map().empty());
							if (!args || args->elements().size() != 1) {
								return Err(
									type_error("_BufferedIOBase.readinto() takes exactly "
											   "one argument ({} given)",
										args->elements().size()));
							}
							auto arg = PyObject::from(args->elements()[0]);
							if (arg.is_err()) return arg;

							PyBuffer buffer;
							int flags = 1;
							return arg.unwrap()
								->get_buffer(buffer, flags)
								.or_else([arg](auto) -> PyResult<std::monostate> {
									return Err(type_error(
										"readinto() argument must be read-write bytes-like "
										"object, not {}",
										arg.unwrap()->type()->name()));
								})
								.and_then(
									[&buffer, arg, self](auto) -> py::PyResult<py::PyObject *> {
										if (buffer.is_ccontiguous()) {
											return self->readinto(buffer);
										} else {
											return Err(type_error(
												"readinto() argument must be a contiguous "
												"buffer, not {}",
												arg.unwrap()->type()->name()));
										}
									});
						})
					.def("readinto1",
						[](BufferedIOBase *self,
							PyTuple *args,
							PyDict *kwargs) -> py::PyResult<py::PyObject *> {
							ASSERT(!kwargs || kwargs->map().empty());
							if (!args || args->elements().size() != 1) {
								return Err(
									type_error("_BufferedIOBase.readinto1() takes exactly "
											   "one argument ({} given)",
										args->elements().size()));
							}
							auto arg = PyObject::from(args->elements()[0]);
							if (arg.is_err()) return arg;

							PyBuffer buffer;
							int flags = 1;
							return arg.unwrap()
								->get_buffer(buffer, flags)
								.or_else([arg](auto) -> PyResult<std::monostate> {
									return Err(type_error(
										"readinto1() argument must be read-write bytes-like "
										"object, not {}",
										arg.unwrap()->type()->name()));
								})
								.and_then(
									[&buffer, arg, self](auto) -> py::PyResult<py::PyObject *> {
										if (buffer.is_ccontiguous()) {
											return self->readinto(buffer);
										} else {
											return Err(type_error(
												"readinto() argument must be a contiguous "
												"buffer, not {}",
												arg.unwrap()->type()->name()));
										}
									});
						})
					.def("write", &BufferedIOBase::write)
					.finalize();
		}
		module->add_symbol(PyString::create("_BufferedIOBase").unwrap(), s_io_buffered_io_base);
		return s_io_buffered_io_base;
	}
};

template<typename T>
// requires(std::is_base_of_v<PyObject, T>)
struct Buffered
{
	PyObject *raw{ nullptr };
	bool ok{ false };
	bool detached{ false };
	bool readable_{ false };
	bool writable_{ false };
	bool finalizing{ false };
	bool fast_closed_checks{ false };
	std::unique_ptr<std::streambuf> buffer;

	bool valid_readbuffer() { return readable_ && buffer && buffer->in_avail() != -1; }

	int64_t readahead() { return valid_readbuffer() ? buffer->in_avail() : 0; }

	PyResult<std::monostate> check_initialized() const
	{
		if (!ok) {
			if (detached) {
				return Err(value_error("raw stream has been detached"));
			} else {
				return Err(value_error("I/O operation on uninitialized object"));
			}
		}

		return Ok(std::monostate{});
	}

	PyResult<PyObject *> detach()
	{
		return static_cast<T *>(this)
			->get_method(PyString::create("flush").unwrap())
			.and_then([](PyObject *flush) {
				return flush->call(PyTuple::create().unwrap(), PyDict::create().unwrap());
			})
			.and_then([this](PyObject *) {
				auto *raw = this->raw;
				this->raw = nullptr;
				this->detached = true;
				this->ok = false;
				return Ok(raw);
			});
	}

	PyResult<PyObject *> simple_flush() const
	{
		if (auto err = check_initialized(); err.is_err()) return Err(err.unwrap_err());

		return raw->get_method(PyString::create("flush").unwrap()).and_then([](PyObject *flush) {
			return flush->call(PyTuple::create().unwrap(), PyDict::create().unwrap());
		});
	}

	PyResult<bool> closed() const
	{
		if (auto err = check_initialized(); err.is_err()) return Err(err.unwrap_err());
		return raw->get_attribute(PyString::create("closed").unwrap())
			.and_then([](PyObject *closed) {
				return truthy(closed, VirtualMachine::the().interpreter());
			});
	}

	PyResult<PyObject *> close()
	{
		if (auto err = check_initialized(); err.is_err()) return Err(err.unwrap_err());

		// FIXME add lock
		auto r = closed();
		if (r.is_err()) return Err(r.unwrap_err());
		if (r.unwrap()) return Ok(py_none());

		if (finalizing) { TODO(); }

		auto res =
			static_cast<T *>(this)
				->get_method(PyString::create("flush").unwrap())
				.and_then([](PyObject *flush) -> PyResult<PyObject *> {
					return flush->call(PyTuple::create().unwrap(), PyDict::create().unwrap());
				});

		res = raw->get_method(PyString::create("close").unwrap())
				  .and_then([](PyObject *close) -> PyResult<PyObject *> {
					  return close->call(PyTuple::create().unwrap(), PyDict::create().unwrap());
				  });

		if (buffer) { buffer = nullptr; }

		return res;
	}

	PyResult<PyObject *> seekable() const
	{
		if (auto err = check_initialized(); err.is_err()) return Err(err.unwrap_err());
		return static_cast<const T *>(this)
			->get_method(PyString::create("seekable").unwrap())
			.and_then([](PyObject *seekable) -> PyResult<PyObject *> {
				return seekable->call(PyTuple::create().unwrap(), PyDict::create().unwrap());
			});
	}

	PyResult<PyObject *> writable() const
	{
		if (auto err = check_initialized(); err.is_err()) return Err(err.unwrap_err());
		return static_cast<const T *>(this)
			->get_method(PyString::create("writable").unwrap())
			.and_then([](PyObject *writable) -> PyResult<PyObject *> {
				return writable->call(PyTuple::create().unwrap(), PyDict::create().unwrap());
			});
	}

	PyResult<PyObject *> readable() const
	{
		if (auto err = check_initialized(); err.is_err()) return Err(err.unwrap_err());
		return static_cast<const T *>(this)
			->get_method(PyString::create("readable").unwrap())
			.and_then([](PyObject *readable) -> PyResult<PyObject *> {
				return readable->call(PyTuple::create().unwrap(), PyDict::create().unwrap());
			});
	}

	PyResult<PyObject *> fileno() const
	{
		if (auto err = check_initialized(); err.is_err()) return Err(err.unwrap_err());
		return static_cast<const T *>(this)
			->get_method(PyString::create("fileno").unwrap())
			.and_then([](PyObject *fileno) -> PyResult<PyObject *> {
				return fileno->call(PyTuple::create().unwrap(), PyDict::create().unwrap());
			});
	}

	PyResult<PyObject *> isatty() const
	{
		if (auto err = check_initialized(); err.is_err()) return Err(err.unwrap_err());
		return static_cast<const T *>(this)
			->get_method(PyString::create("isatty").unwrap())
			.and_then([](PyObject *isatty) -> PyResult<PyObject *> {
				return isatty->call(PyTuple::create().unwrap(), PyDict::create().unwrap());
			});
	}

	PyResult<PyObject *> _dealloc_warn(PyObject *source) const
	{
		if (this->ok && this->raw) {
			this->raw->get_method(PyString::create("_dealloc_warn").unwrap())
				.and_then([source](PyObject *_dealloc_warn) {
					return _dealloc_warn->call(
						PyTuple::create(source).unwrap(), PyDict::create().unwrap());
				});
		}
		return Ok(py_none());
	}

	bool is_closed() const
	{
		if (!buffer) return false;
		if (this->fast_closed_checks) {
			TODO();
		} else {
			return closed().or_else([](auto) { return Ok(false); }).unwrap();
		}
	}

	PyResult<PyObject *> read(int64_t n)
	{
		if (auto err = check_initialized(); err.is_err()) return Err(err.unwrap_err());

		if (n < -1) { return Err(value_error("read length must be non-negative or -1")); }

		if (is_closed()) { return Err(value_error("read of closed file")); }

		if (n == -1) {
			return static_cast<T *>(this)->readall();
		} else {
			auto res = static_cast<T *>(this)->readfast(n);
			if (res.is_ok() && res.unwrap() != py_none()) { return res; }
			return static_cast<T *>(this)->readgeneric(n);
		}
	}

	void visit_graph_buffered(Cell::Visitor &visitor)
	{
		if (raw) { visitor.visit(*raw); }
	}
};

class BufferedReader
	: public BufferedIOBase
	, public Buffered<BufferedReader>
{
	friend class ::Heap;

	BufferedReader(PyType *type, PyObject *raw, int buffer_size) : BufferedIOBase(type)
	{
		this->raw = raw;
		(void)buffer_size;
		this->readable_ = true;
		this->writable_ = false;
		this->fast_closed_checks = false;
		this->ok = true;
	}

	BufferedReader(PyType *type) : BufferedReader(type, nullptr, 0) {}

  protected:
	static PyResult<BufferedReader *> create(PyType *type, PyObject *raw, int buffer_size)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<BufferedReader>(type, raw, buffer_size)) { return Ok(obj); }
		return Err(memory_error(sizeof(BufferedReader)));
	}

	static PyResult<BufferedReader *> create(PyType *type)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<BufferedReader>(type)) { return Ok(obj); }
		return Err(memory_error(sizeof(BufferedReader)));
	}

  public:
	static constexpr std::string_view __doc__ =
		"BufferedReader(raw, buffer_size=DEFAULT_BUFFER_SIZE)\n"
		"--\n"
		"\n"
		"Create a new buffered reader using the given readable raw IO object.";

	static PyResult<BufferedReader *> create(PyObject *raw, int buffer_size)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<BufferedReader>(s_io_buffered_reader, raw, buffer_size)) {
			return Ok(obj);
		}
		return Err(memory_error(sizeof(BufferedReader)));
	}

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *, PyDict *)
	{
		return BufferedReader::create(const_cast<PyType *>(type));
	}

	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs)
	{
		ASSERT(!kwargs || kwargs->map().empty());
		if (!args || args->elements().empty()) {
			return Err(type_error("missing required argument 'raw'"));
		}
		if (args->elements().size() > 1) { TODO(); }
		return PyObject::from(args->elements()[0])
			.and_then([this](PyObject *raw) -> PyResult<int32_t> {
				this->raw = raw;
				this->readable_ = true;
				this->writable_ = false;
				this->fast_closed_checks = false;
				this->ok = true;
				this->buffer = std::make_unique<std::stringbuf>();
				return Ok(0);
			});
	}

	PyResult<PyObject *> readall()
	{
		std::vector<std::byte> data;

		// copy out all the data in the current read buffer
		auto current_size = readahead();
		if (current_size > 0) {
			ASSERT(buffer);
			data.resize(current_size);
			buffer->sgetn(::bit_cast<char *>(data.begin().base()), current_size);
		}

		auto readall_ = raw->lookup_attribute(PyString::create("readall").unwrap());
		if (std::get<0>(readall_).is_err()) std::get<0>(readall_);
		auto *readall_obj = std::get<0>(readall_).unwrap();

		// readall from raw
		if (readall_obj != py_none()) {
			auto tmp = readall_obj->call(PyTuple::create().unwrap(), PyDict::create().unwrap());
			if (tmp.is_err()) return tmp;
			if (tmp.unwrap() != py_none() && !as<PyBytes>(tmp.unwrap())) {
				return Err(type_error(
					"readall() should return bytes or None, not {}", tmp.unwrap()->type()->name()));
			}
			if (current_size == 0) {
				return tmp;
			} else if (tmp.unwrap() != py_none()) {
				const auto &bytes = as<PyBytes>(tmp.unwrap())->value().b;
				data.insert(data.end(), bytes.begin(), bytes.end());
			}
			return PyBytes::create(Bytes{ std::move(data) });
		}

		// no readall implementation provided so call read until the end
		while (true) {
			auto new_data =
				raw->get_method(PyString::create("read").unwrap()).and_then([](PyObject *read) {
					return read->call(PyTuple::create().unwrap(), PyDict::create().unwrap());
				});
			if (new_data.is_err()) return new_data;
			if (new_data.unwrap() != py_none() && !as<PyBytes>(new_data.unwrap())) {
				return Err(type_error("read() should return bytes or None, not {}",
					new_data.unwrap()->type()->name()));
			}
			if (new_data.unwrap() == py_none()
				|| as<PyBytes>(new_data.unwrap())->value().b.empty()) {
				if (current_size == 0) {
					return new_data;
				} else if (new_data.unwrap() != py_none()) {
					const auto &bytes = as<PyBytes>(new_data.unwrap())->value().b;
					data.insert(data.end(), bytes.begin(), bytes.end());
				}
				return PyBytes::create(Bytes{ std::move(data) });
			} else {
				const auto &bytes = as<PyBytes>(new_data.unwrap())->value().b;
				current_size += bytes.size();
				data.insert(data.end(), bytes.begin(), bytes.end());
			}
		}
	}

	PyResult<PyObject *> readfast(size_t n)
	{
		const auto current_size = readahead();
		if (static_cast<int64_t>(n) <= current_size) {
			ASSERT(buffer);
			std::vector<std::byte> data;
			data.resize(current_size);
			buffer->sgetn(::bit_cast<char *>(data.begin().base()), current_size);
			return PyBytes::create(Bytes{ std::move(data) });
		}
		return Ok(py_none());
	}

	PyResult<PyObject *> readgeneric(size_t)
	{
		return Err(not_implemented_error("BufferedReader.readgeneric"));
	}

	PyResult<PyObject *> __repr__() const
	{
		auto attr = lookup_attribute(PyString::create("name").unwrap());
		if (std::get<0>(attr).is_err()
			&& std::get<0>(attr).unwrap_err()->type() != value_error("")->type()) {
			return std::get<0>(attr);
		}
		if (std::get<1>(attr) == LookupAttrResult::NOT_FOUND) {
			return PyString::create(fmt::format("<{}>", type_prototype().__name__));
		}
		auto *nameobj = std::get<0>(attr).unwrap();
		return nameobj->repr().and_then([this](PyObject *nameobj_repr) {
			return PyString::create(
				fmt::format("<{} name={}>", type_prototype().__name__, nameobj_repr->to_string()));
		});
	}

	static PyType *register_type(PyModule *module)
	{
		if (!s_io_buffered_reader) {
			s_io_buffered_reader =
				klass<BufferedReader>(module, "BufferedReader", s_io_buffered_io_base)
					.def("detach", &Buffered<BufferedReader>::detach)
					.def("flush", &Buffered<BufferedReader>::simple_flush)
					.def("close", &Buffered<BufferedReader>::close)
					.def("seekable", &Buffered<BufferedReader>::seekable)
					.def("readable", &Buffered<BufferedReader>::readable)
					.def("fileno", &Buffered<BufferedReader>::fileno)
					.def("isatty", &Buffered<BufferedReader>::isatty)
					.def("_dealloc_warn",
						[](BufferedReader *self,
							PyTuple *args,
							PyDict *kwargs) -> PyResult<PyObject *> {
							ASSERT(!kwargs || kwargs->map().empty());
							if (args->elements().size() != 1) {
								return Err(
									type_error("BufferedReader._dealloc_warn() takes exactly one "
											   "argument ({} given)",
										args->elements().size()));
							}
							return PyObject::from(args->elements()[0])
								.and_then([self](PyObject *source) {
									return self->_dealloc_warn(source);
								});
						})
					.def("read",
						[](BufferedReader *self,
							PyTuple *args,
							PyDict *kwargs) -> PyResult<PyObject *> {
							ASSERT(!kwargs || kwargs->map().empty());
							int64_t n = -1;
							if (args && args->elements().size() > 1) {
								return Err(value_error(
									"BaseIO.readlines expected at most one argument (got {})",
									args->elements().size()));
							} else if (args && args->elements().size() == 1) {
								auto arg0 = PyObject::from(args->elements()[0]);
								if (arg0.is_err()) return arg0;
								if (!as<PyInteger>(arg0.unwrap()) && arg0.unwrap() != py_none()) {
									return Err(
										type_error("argument should be integer or None, not '{}'",
											arg0.unwrap()->type()->name()));
								}
								if (arg0.unwrap() != py_none()) {
									n = as<PyInteger>(arg0.unwrap())->as_i64();
								}
							}
							return static_cast<Buffered<BufferedReader> *>(self)->read(n);
						})
					// .def("peek", &BufferedReader::peek)
					// .def("read1", &BufferedReader::read1)
					// .def("readinto", &BufferedReader::readinto)
					// .def("readinto1", &BufferedReader::readinto1)
					// .def("readline", &BufferedReader::readline)
					// .def("seek", &BufferedReader::seek)
					// .def("tell", &Buffered::tell)
					// .def("truncate", &BufferedReader::truncate)
					// .def("__sizeof__", &Buffered::__sizeof__)
					// .attr("raw", &Buffered::raw)
					// .attr("_finalizing", &Buffered::finalizing)
					// .property_readonly("closed")
					.property_readonly("name",
						[](BufferedReader *self) {
							if (self->check_initialized().is_err()) { TODO(); }
							ASSERT(self->raw);
							auto attr = self->raw->get_attribute(PyString::create("name").unwrap());
							return attr;
						})
					// .property_readonly("mode")
					.finalize();
		}
		module->add_symbol(PyString::create("BufferedReader").unwrap(), s_io_buffered_reader);
		return s_io_buffered_reader;
	}

	PyType *static_type() const override { return s_io_buffered_reader; }

	static PyType *class_type() { return s_io_buffered_reader; }

	void visit_graph(Visitor &visitor) override
	{
		PyObject::visit_graph(visitor);
		visit_graph_buffered(visitor);
	}
};

class BufferedWriter
	: public BufferedIOBase
	, public Buffered<BufferedWriter>
{
	friend class ::Heap;

	BufferedWriter(PyType *type, PyObject *raw, int buffer_size) : BufferedIOBase(type)
	{
		this->raw = raw;
		(void)buffer_size;
		this->readable_ = false;
		this->writable_ = true;
		this->fast_closed_checks = false;
		this->ok = true;
	}

	BufferedWriter(PyType *type) : BufferedWriter(type, nullptr, 0) {}

  protected:
	static PyResult<BufferedWriter *> create(PyType *type, PyObject *raw, int buffer_size)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<BufferedWriter>(type, raw, buffer_size)) { return Ok(obj); }
		return Err(memory_error(sizeof(BufferedWriter)));
	}

	static PyResult<BufferedWriter *> create(PyType *type)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<BufferedWriter>(type)) { return Ok(obj); }
		return Err(memory_error(sizeof(BufferedWriter)));
	}

  public:
	static constexpr std::string_view __doc__ =
		"BufferedWriter(raw, buffer_size=DEFAULT_BUFFER_SIZE)\n"
		"--\n"
		"\n"
		"A buffer for a writeable sequential RawIO object.\n"
		"\n"
		"The constructor creates a BufferedWriter for the given writeable raw\n"
		"stream. If the buffer_size is not given, it defaults to\n"
		"DEFAULT_BUFFER_SIZE.";

	static PyResult<BufferedWriter *> create(PyObject *raw, int buffer_size)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<BufferedWriter>(s_io_buffered_reader, raw, buffer_size)) {
			return Ok(obj);
		}
		return Err(memory_error(sizeof(BufferedWriter)));
	}

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *, PyDict *)
	{
		return BufferedWriter::create(const_cast<PyType *>(type));
	}

	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs)
	{
		ASSERT(!kwargs || kwargs->map().empty());
		if (!args || args->elements().empty()) {
			return Err(type_error("missing required argument 'raw'"));
		}
		if (args->elements().size() > 1) { TODO(); }
		return PyObject::from(args->elements()[0])
			.and_then([this](PyObject *raw) -> PyResult<int32_t> {
				this->raw = raw;
				this->readable_ = false;
				this->writable_ = true;
				this->fast_closed_checks = false;
				this->ok = true;
				this->buffer = std::make_unique<std::stringbuf>();
				return Ok(0);
			});
	}

	PyResult<PyObject *> __repr__() const
	{
		auto attr = lookup_attribute(PyString::create("name").unwrap());
		if (std::get<0>(attr).is_err()
			&& std::get<0>(attr).unwrap_err()->type() != value_error("")->type()) {
			return std::get<0>(attr);
		}
		if (std::get<1>(attr) == LookupAttrResult::NOT_FOUND) {
			return PyString::create(fmt::format("<{}>", type_prototype().__name__));
		}
		auto *nameobj = std::get<0>(attr).unwrap();
		return nameobj->repr().and_then([this](PyObject *nameobj_repr) {
			return PyString::create(
				fmt::format("<{} name={}>", type_prototype().__name__, nameobj_repr->to_string()));
		});
	}

	static PyType *register_type(PyModule *module)
	{
		if (!s_io_buffered_writer) {
			s_io_buffered_writer =
				klass<BufferedWriter>(module, "BufferedWriter", s_io_buffered_io_base).finalize();
		}
		module->add_symbol(PyString::create("BufferedWriter").unwrap(), s_io_buffered_writer);
		return s_io_buffered_writer;
	}

	PyType *static_type() const override { return s_io_buffered_writer; }

	static PyType *class_type() { return s_io_buffered_writer; }

	void visit_graph(Visitor &visitor) override
	{
		PyObject::visit_graph(visitor);
		visit_graph_buffered(visitor);
	}
};

template<> BufferedReader *as(PyObject *obj)
{
	if (obj->type() == BufferedReader::class_type()) { return static_cast<BufferedReader *>(obj); }
	return nullptr;
}

template<> const BufferedReader *as(const PyObject *obj)
{
	if (obj->type() == BufferedReader::class_type()) {
		return static_cast<const BufferedReader *>(obj);
	}
	return nullptr;
}

template<> BufferedWriter *as(PyObject *obj)
{
	if (obj->type() == BufferedWriter::class_type()) { return static_cast<BufferedWriter *>(obj); }
	return nullptr;
}

template<> const BufferedWriter *as(const PyObject *obj)
{
	if (obj->type() == BufferedWriter::class_type()) {
		return static_cast<const BufferedWriter *>(obj);
	}
	return nullptr;
}

class BufferedRWPair : public BufferedIOBase
{
	friend class ::Heap;
	BufferedReader *m_reader;
	BufferedWriter *m_writer;
	size_t m_buffer_size;

	BufferedRWPair(PyType *type) : BufferedIOBase(type) {}

  protected:
	static PyResult<BufferedRWPair *> create(PyType *type)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<BufferedRWPair>(type)) { return Ok(obj); }
		return Err(memory_error(sizeof(BufferedRWPair)));
	}

  public:
	static constexpr std::string_view __doc__ =
		"BufferedRWPair(reader, writer, buffer_size=DEFAULT_BUFFER_SIZE, /)\n"
		"--\n"
		"\n"
		"A buffered reader and writer object together.\n"
		"\n"
		"A buffered reader object and buffered writer object put together to\n"
		"form a sequential IO object that can read and write. This is typically\n"
		"used with a socket or two-way pipe.\n"
		"\n"
		"reader and writer are RawIOBase objects that are readable and\n"
		"writeable respectively. If the buffer_size is omitted it defaults to\n"
		"DEFAULT_BUFFER_SIZE.";

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *, PyDict *)
	{
		return BufferedRWPair::create(const_cast<PyType *>(type));
	}

	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs)
	{
		if (this->type() != s_io_buffered_rwpair && kwargs && !kwargs->map().empty()) {
			return Err(type_error("BufferedRWPair() takes no keyword arguments"));
		}

		auto parse_result = PyArgsParser<PyObject *, PyObject *, PyInteger *>::unpack_tuple(args,
			kwargs,
			"BufferedRWPair",
			std::integral_constant<size_t, 2>{},
			std::integral_constant<size_t, 3>{},
			PyInteger::create(8 * 1024).unwrap());

		if (parse_result.is_err()) return Err(parse_result.unwrap_err());
		auto [reader, writer, buffer_size] = parse_result.unwrap();

		if (auto readable = check_readable(reader); readable.is_err()) {
			return Err(readable.unwrap_err());
		}
		if (auto writable = check_writable(reader); writable.is_err()) {
			return Err(writable.unwrap_err());
		}

		auto reader_ = BufferedReader::class_type()->call(
			PyTuple::create(reader, buffer_size).unwrap(), nullptr);
		if (reader_.is_err()) return Err(reader_.unwrap_err());
		m_reader = as<BufferedReader>(reader_.unwrap());
		ASSERT(m_reader);

		auto writer_ = BufferedWriter::class_type()->call(
			PyTuple::create(writer, buffer_size).unwrap(), nullptr);
		if (writer_.is_err()) return Err(writer_.unwrap_err());
		m_writer = as<BufferedWriter>(writer_.unwrap());
		ASSERT(m_writer);

		m_buffer_size = buffer_size->as_size_t();

		return Ok(0);
	}

	PyResult<PyObject *> read(PyTuple *args, PyDict *)
	{
		return m_reader->get_method(PyString::create("read").unwrap()).and_then([args](auto *read) {
			return read->call(args, nullptr);
		});
	}

	PyResult<PyObject *> peek(PyTuple *args, PyDict *)
	{
		return m_reader->get_method(PyString::create("peek").unwrap()).and_then([args](auto *peek) {
			return peek->call(args, nullptr);
		});
	}

	PyResult<PyObject *> read1(PyTuple *args, PyDict *)
	{
		return m_reader->get_method(PyString::create("read1").unwrap())
			.and_then([args](auto *read1) { return read1->call(args, nullptr); });
	}

	PyResult<PyObject *> readinto(PyTuple *args, PyDict *)
	{
		return m_reader->get_method(PyString::create("readinto").unwrap())
			.and_then([args](auto *readinto) { return readinto->call(args, nullptr); });
	}

	PyResult<PyObject *> write(PyTuple *args, PyDict *)
	{
		return m_writer->get_method(PyString::create("write").unwrap())
			.and_then([args](auto *write) { return write->call(args, nullptr); });
	}

	PyResult<PyObject *> flush(PyTuple *, PyDict *)
	{
		return m_writer->get_method(PyString::create("flush").unwrap()).and_then([](auto *write) {
			return write->call(nullptr, nullptr);
		});
	}

	PyResult<PyObject *> readable(PyTuple *, PyDict *)
	{
		return m_reader->get_method(PyString::create("readable").unwrap())
			.and_then([](auto *readable) { return readable->call(nullptr, nullptr); });
	}

	PyResult<PyObject *> writable(PyTuple *, PyDict *)
	{
		return m_reader->get_method(PyString::create("writable").unwrap())
			.and_then([](auto *writable) { return writable->call(nullptr, nullptr); });
	}

	PyResult<PyObject *> close(PyTuple *, PyDict *)
	{
		auto writer_result =
			m_reader->get_method(PyString::create("close").unwrap()).and_then([](auto *close) {
				return close->call(nullptr, nullptr);
			});

		auto reader_result =
			m_reader->get_method(PyString::create("close").unwrap()).and_then([](auto *close) {
				return close->call(nullptr, nullptr);
			});

		if (writer_result.is_err()) {
			// TODO: chain reader and writer exceptions
			return writer_result;
		}

		return reader_result;
	}

	PyResult<PyObject *> isatty(PyTuple *, PyDict *)
	{
		return m_writer->get_method(PyString::create("isatty").unwrap())
			.and_then([](auto *isatty) { return isatty->call(nullptr, nullptr); })
			.and_then([this](auto *result) -> PyResult<PyObject *> {
				if (result == py_false()) {
					return m_reader->get_method(PyString::create("isatty").unwrap())
						.and_then([](auto *isatty) { return isatty->call(nullptr, nullptr); });
				}
				return Ok(result);
			});
	}

	static PyType *register_type(PyModule *module)
	{
		if (!s_io_buffered_rwpair) {
			s_io_buffered_rwpair =
				klass<BufferedRWPair>(module, "BufferedRWPair", s_io_buffered_io_base).finalize();
		}
		module->add_symbol(PyString::create("BufferedRWPair").unwrap(), s_io_buffered_rwpair);
		return s_io_buffered_rwpair;
	}

	PyType *static_type() const override { return s_io_buffered_rwpair; }

	void visit_graph(Visitor &visitor) override
	{
		PyObject::visit_graph(visitor);
		if (m_reader) m_reader->visit_graph(visitor);
		if (m_writer) m_writer->visit_graph(visitor);
	}
};


class BufferedRandom
	: public BufferedIOBase
	, public Buffered<BufferedRandom>
{
	friend class ::Heap;

	BufferedRandom(PyType *type, PyObject *raw, int buffer_size) : BufferedIOBase(type)
	{
		this->raw = raw;
		(void)buffer_size;
		this->readable_ = true;
		this->writable_ = true;
		this->detached = true;
		this->fast_closed_checks = false;
		this->ok = false;
	}

	BufferedRandom(PyType *type) : BufferedRandom(type, nullptr, 0) {}

  protected:
	static PyResult<BufferedRandom *> create(PyType *type, PyObject *raw, int buffer_size)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<BufferedRandom>(type, raw, buffer_size)) { return Ok(obj); }
		return Err(memory_error(sizeof(BufferedRandom)));
	}

	static PyResult<BufferedRandom *> create(PyType *type)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<BufferedRandom>(type)) { return Ok(obj); }
		return Err(memory_error(sizeof(BufferedRandom)));
	}

  public:
	static constexpr std::string_view __doc__ =
		"BufferedRandom(raw, buffer_size=DEFAULT_BUFFER_SIZE)\n"
		"--\n"
		"\n"
		"A buffered interface to random access streams.\n"
		"\n"
		"The constructor creates a reader and writer for a seekable stream,\n"
		"raw, given in the first argument. If the buffer_size is omitted it\n"
		"defaults to DEFAULT_BUFFER_SIZE.";

	static PyResult<BufferedRandom *> create(PyObject *raw, int buffer_size)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<BufferedRandom>(s_io_buffered_reader, raw, buffer_size)) {
			return Ok(obj);
		}
		return Err(memory_error(sizeof(BufferedRandom)));
	}

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *, PyDict *)
	{
		return BufferedRandom::create(const_cast<PyType *>(type));
	}

	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs)
	{
		auto parse_result = PyArgsParser<PyObject *, PyInteger *>::unpack_tuple(args,
			kwargs,
			"BufferedRandom",
			std::integral_constant<size_t, 1>{},
			std::integral_constant<size_t, 2>{},
			PyInteger::create(8 * 1024).unwrap());

		if (parse_result.is_err()) return Err(parse_result.unwrap_err());
		auto [raw, buffer_size] = parse_result.unwrap();

		if (auto seekable = check_seekable(raw); seekable.is_err()) {
			return Err(seekable.unwrap_err());
		}
		if (auto readable = check_readable(raw); readable.is_err()) {
			return Err(readable.unwrap_err());
		}
		if (auto writable = check_writable(raw); writable.is_err()) {
			return Err(writable.unwrap_err());
		}

		this->raw = raw;

		this->raw = raw;
		this->readable_ = true;
		this->writable_ = true;
		this->fast_closed_checks = false;
		this->ok = true;
		this->buffer = std::make_unique<std::stringbuf>();

		return Ok(0);
	}

	PyResult<PyObject *> __repr__() const
	{
		auto attr = lookup_attribute(PyString::create("name").unwrap());
		if (std::get<0>(attr).is_err()
			&& std::get<0>(attr).unwrap_err()->type() != value_error("")->type()) {
			return std::get<0>(attr);
		}
		if (std::get<1>(attr) == LookupAttrResult::NOT_FOUND) {
			return PyString::create(fmt::format("<{}>", type_prototype().__name__));
		}
		auto *nameobj = std::get<0>(attr).unwrap();
		return nameobj->repr().and_then([this](PyObject *nameobj_repr) {
			return PyString::create(
				fmt::format("<{} name={}>", type_prototype().__name__, nameobj_repr->to_string()));
		});
	}

	static PyType *register_type(PyModule *module)
	{
		if (!s_io_buffered_random) {
			s_io_buffered_random =
				klass<BufferedRandom>(module, "BufferedRandom", s_io_buffered_io_base)
					.def("close", &Buffered<BufferedRandom>::close)
					.def("detach", &Buffered<BufferedRandom>::detach)
					.def("seekable", &Buffered<BufferedRandom>::seekable)
					.def("readable", &Buffered<BufferedRandom>::readable)
					.def("writable", &Buffered<BufferedRandom>::writable)
					.def("fileno", &Buffered<BufferedRandom>::fileno)
					.def("isatty", &Buffered<BufferedRandom>::isatty)
					.finalize();
		}
		module->add_symbol(PyString::create("BufferedRandom").unwrap(), s_io_buffered_random);
		return s_io_buffered_random;
	}

	PyType *static_type() const override { return s_io_buffered_random; }

	static PyType *class_type() { return s_io_buffered_random; }

	void visit_graph(Visitor &visitor) override
	{
		PyObject::visit_graph(visitor);
		visit_graph_buffered(visitor);
	}
};

class BytesIO : public BufferedIOBase
{
	friend ::Heap;
	PyObject *m_buf;
	int64_t m_pos;
	int64_t m_string_size;

	BytesIO(PyType *type) : BufferedIOBase(type) {}

  public:
	static constexpr std::string_view __doc__ =
		"BytesIO(initial_bytes=b\'\')\n"
		"--\n"
		"\n"
		"Buffered I/O implementation using an in-memory bytes buffer.";

	static PyResult<PyObject *> create(PyType *type)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<BytesIO>(type)) { return Ok(obj); }
		return Err(memory_error(sizeof(BytesIO)));
	}

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *, PyDict *)
	{
		return BytesIO::create(const_cast<PyType *>(type));
	}

	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs)
	{
		ASSERT(!kwargs || kwargs->map().empty());
		m_pos = 0;
		m_string_size = 0;
		if (args && args->elements().size() == 1) {
			return PyObject::from(args->elements()[0]).and_then([this](PyObject *initial_bytes) {
				if (as<PyBytes>(initial_bytes)) {
					m_buf = initial_bytes;
					m_string_size = as<PyBytes>(initial_bytes)->value().b.size();
					return Ok(0);
				} else if (initial_bytes == py_none()) {
					return Ok(0);
				} else {
					TODO();
				}
			});
		}

		return PyBytes::create(Bytes{}).and_then([this](auto *initial_bytes) -> PyResult<int32_t> {
			m_buf = initial_bytes;
			return Ok(0);
		});
	}

	PyResult<std::monostate> check_closed() const
	{
		if (!m_buf) { return Err(value_error("I/O operation on closed file.")); }
		return Ok(std::monostate{});
	}

	PyResult<PyObject *> read(int64_t n)
	{
		ASSERT(m_buf);
		ASSERT(as<PyBytes>(m_buf));

		if (auto err = check_closed(); err.is_err()) return Err(err.unwrap_err());

		const auto size = m_string_size - m_pos;

		n = n < 0 ? size : n;
		n = std::clamp(n, int64_t{ 0 }, size);

		if (n > 1 && m_pos == 0
			&& n == static_cast<int64_t>(as<PyBytes>(m_buf)->value().b.size())) {
			m_pos += n;
			return Ok(m_buf);
		}

		std::vector<std::byte> output;
		const auto &bytes = as<PyBytes>(m_buf)->value().b;
		output.insert(output.end(), bytes.begin() + m_pos, bytes.begin() + m_pos + n);
		m_pos += n;
		return PyBytes::create(Bytes{ std::move(output) });
	}

	PyResult<PyObject *> read1(int64_t n) { return read(n); }

	PyType *static_type() const override { return s_io_bytesio; }

	static PyType *register_type(PyModule *module)
	{
		if (!s_io_bytesio) {
			s_io_bytesio =
				klass<BytesIO>(module, "BytesIO", s_io_buffered_io_base)
					.def("read",
						[](BytesIO *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
							ASSERT(!kwargs || kwargs->map().empty());
							int64_t n = -1;
							if (args && args->elements().size() > 1) {
								return Err(value_error(
									"BytesIO.readlines expected at most one argument (got {})",
									args->elements().size()));
							} else if (args && args->elements().size() == 1) {
								auto arg0 = PyObject::from(args->elements()[0]);
								if (arg0.is_err()) return arg0;
								if (!as<PyInteger>(arg0.unwrap()) && arg0.unwrap() != py_none()) {
									return Err(
										type_error("argument should be integer or None, not '{}'",
											arg0.unwrap()->type()->name()));
								}
								if (arg0.unwrap() != py_none()) {
									n = as<PyInteger>(arg0.unwrap())->as_i64();
								}
							}
							return self->read(n);
						})
					.def("read1",
						[](BytesIO *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
							ASSERT(!kwargs || kwargs->map().empty());
							int64_t n = -1;
							if (args && args->elements().size() > 1) {
								return Err(value_error(
									"BytesIO.readlines expected at most one argument (got {})",
									args->elements().size()));
							} else if (args && args->elements().size() == 1) {
								auto arg0 = PyObject::from(args->elements()[0]);
								if (arg0.is_err()) return arg0;
								if (!as<PyInteger>(arg0.unwrap()) && arg0.unwrap() != py_none()) {
									return Err(
										type_error("argument should be integer or None, not '{}'",
											arg0.unwrap()->type()->name()));
								}
								if (arg0.unwrap() != py_none()) {
									n = as<PyInteger>(arg0.unwrap())->as_i64();
								}
							}
							return self->read1(n);
						})

					.finalize();
		}
		module->add_symbol(PyString::create("BytesIO").unwrap(), s_io_bytesio);
		return s_io_bytesio;
	}
};

#if defined(__GLIBCXX__) || defined(__GLIBCPP__)
// taken from https://stackoverflow.com/a/19749019
typedef std::basic_ofstream<char>::__filebuf_type buffer_t;
typedef __gnu_cxx::stdio_filebuf<char> io_buffer_t;
FILE *cfile_impl(buffer_t *const fb)
{
	return (static_cast<io_buffer_t *const>(fb))->file();// type std::__c_file
}

FILE *cfile(std::fstream const &fs) { return cfile_impl(fs.rdbuf()); }
#else
// FIXME: find a way to get the file descriptor using libc++
FILE *cfile(std::fstream const &) { return nullptr; }
#endif

enum Mode {
	CREATE = 0,
	READ = 1,
	WRITE = 2,
	APPEND = 3,
	UPDATE = 4,
	TEXT = 5,
	BINARY = 6,
	UNIVERSAL = 7,
};


PyResult<std::bitset<8>> read_flags(const std::string &mode)
{
	std::bitset<8> flag;

	for (size_t i = 0; const auto c : mode) {
		if (std::string_view{ mode }.substr(0, i).find(c) != std::string::npos) {
			// duplicate mode
			return Err(value_error("invalid mode: '{}'", mode));
		}
		if (c == 'x') {
			flag.set(Mode::CREATE);
		} else if (c == 'r') {
			flag.set(Mode::READ);
		} else if (c == 'w') {
			flag.set(Mode::WRITE);
		} else if (c == 'a') {
			flag.set(Mode::APPEND);
		} else if (c == '+') {
			flag.set(Mode::UPDATE);
		} else if (c == 't') {
			flag.set(Mode::TEXT);
		} else if (c == 'b') {
			flag.set(Mode::BINARY);
		} else if (c == 'U') {
			flag.set(Mode::UNIVERSAL);
			flag.set(Mode::READ);
		} else {
			return Err(value_error("invalid mode: '{}'", mode));
		}
		i++;
	}

	return Ok(flag);
}


class FileIO : public RawIOBase
{
	friend ::Heap;

	int m_file_descriptor{ -1 };
	std::fstream m_filestream;
	bool m_created{ false };
	bool m_readable{ false };
	bool m_writable{ false };
	bool m_appending{ false };
	std::optional<bool> m_seekable;
	bool m_close_file_descriptor{ true };
	bool m_finalizing;
	uint32_t m_bulksize{ 0 };

	FileIO(const PyType *type) : RawIOBase(type) {}

  public:
	static constexpr std::string_view __doc__ =
		"FileIO(file, mode=\'r\', closefd=True, opener=None)\n"
		"--\n"
		"\n"
		"Open a file.\n"
		"\n"
		"The mode can be \'r\' (default), \'w\', \'x\' or \'a\' for reading,\n"
		"writing, exclusive creation or appending.  The file will be created if it\n"
		"doesn\'t exist when opened for writing or appending; it will be truncated\n"
		"when opened for writing.  A FileExistsError will be raised if it already\n"
		"exists when opened for creating. Opening a file for creating implies\n"
		"writing so this mode behaves in a similar way to \'w\'.Add a \'+\' to the mode\n"
		"to allow simultaneous reading and writing. A custom opener can be used by\n"
		"passing a callable as *opener*. The underlying file descriptor for the file\n"
		"object is then obtained by calling opener with (*name*, *flags*).\n"
		"*opener* must return an open file descriptor (passing os.open as *opener*\n"
		"results in functionality similar to passing None).";

	static PyResult<FileIO *>
		create(PyObject *file, std::bitset<8> rawmode, bool close, PyObject *opener)
	{
		(void)close;
		(void)opener;
		return FileIO::__new__(s_io_fileio, nullptr, nullptr)
			.and_then([file, &rawmode](PyObject *fileio) -> PyResult<FileIO *> {
				auto result = static_cast<FileIO *>(fileio)->init(file, rawmode);
				if (result.is_err()) return Err(result.unwrap_err());
				if (result.unwrap() != 0) { TODO(); }
				return Ok(static_cast<FileIO *>(fileio));
			});
	}

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *, PyDict *)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<FileIO>(type)) { return Ok(obj); }
		return Err(memory_error(sizeof(FileIO)));
	}


	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs)
	{
		ASSERT(!kwargs || kwargs->map().empty());
		if (args->elements().size() != 2) { TODO(); }
		auto *filename = PyObject::from(args->elements()[0]).unwrap();
		auto *mode_ = PyObject::from(args->elements()[1]).unwrap();
		if (!as<PyString>(filename)) { TODO(); }
		if (!as<PyString>(mode_)) { TODO(); }

		return read_flags(as<PyString>(mode_)->value())
			.and_then([this, filename](const auto &flags) { return init(filename, flags); });
	}

	std::string to_string() const override
	{
		if (!m_filestream.is_open()) { return "<_io.FileIO [closed]>"; }
		return fmt::format("<_io.FileIO fd={} mode={}>", m_file_descriptor, mode_string());
	}

	PyResult<PyObject *> __repr__() const { return PyString::create(to_string()); }

	PyType *static_type() const override { return s_io_fileio; }

	PyResult<PyObject *> readall()
	{
		if (!m_filestream.is_open()) { return Err(value_error("I/O operation on closed file")); }

		m_filestream.seekg(0);

		if (m_filestream.fail()) { TODO(); }
		const auto initial_position = m_filestream.tellg();
		if (initial_position == -1) { TODO(); }

		m_filestream.seekg(0, std::ios_base::end);
		if (m_filestream.fail()) { TODO(); }
		const auto end_position = m_filestream.tellg();
		if (end_position == -1) { TODO(); }

		m_filestream.seekg(0);
		if (m_filestream.fail()) { TODO(); }

		const auto file_size = end_position - initial_position;
		std::vector<std::byte> result;
		result.resize(file_size);
		int64_t bytes_read = 0;
		const auto buffer_size = file_size;

		do {
			if (m_filestream.fail()) { TODO(); }
			m_filestream.read(::bit_cast<char *>(result.data()), buffer_size);
			bytes_read += m_filestream.gcount();
		} while (!m_filestream.eof());

		// we expect failbit to be set when we read up to eof, but not badbit to be set
		if (m_filestream.rdstate() & std::ios_base::badbit) { TODO(); }
		// we should always reach the end of the file, otherwise something went wrong
		if (bytes_read != file_size) { TODO(); }

		// set to eof, which removes the failbit
		m_filestream.clear(std::ios_base::eofbit);
		// make sure that we are not failing anymore
		if (m_filestream.fail()) { TODO(); }

		return PyBytes::create(Bytes{ std::move(result) });
	}

	PyResult<PyObject *> close()
	{
		// RawIOBase::close(this);
		if (m_filestream.fail()) { TODO(); }
		if (!m_filestream.is_open()) { return Ok(py_none()); }
		m_filestream.close();
		if (m_filestream.fail()) { TODO(); }
		return Ok(py_none());
	}

	static PyType *register_type(PyModule *module)
	{
		if (!s_io_fileio) {
			s_io_fileio = klass<FileIO>(module, "FileIO", s_io_raw_iobase)
							  .def("readall", &FileIO::readall)
							  .def("close", &FileIO::close)
							  .finalize();
		}
		module->add_symbol(PyString::create("FileIO").unwrap(), s_io_fileio);
		return s_io_fileio;
	}

  private:
	PyResult<int32_t> init(PyObject *filename, const std::bitset<8> &rawmode)
	{
		auto mode_result = get_mode(rawmode);
		if (mode_result.is_err()) return Err(mode_result.unwrap_err());
		const auto mode = mode_result.unwrap();


		const fs::path filepath = as<PyString>(filename)->value();
		m_filestream.open(filepath, mode);
		if (m_filestream.fail()) {
			// FIXME: can we avoid using errno, and figure out the error using fs::perms
			auto *msg = strerror(errno);
			// FIXME: should be OSError
			return Err(value_error("{}", msg));
		}

		if (auto *file_ptr = cfile(m_filestream)) {
#if defined(__linux__)
			m_file_descriptor = file_ptr->_fileno;
#elif defined(__APPLE__)
			m_file_descriptor = file_ptr->_file;
#else
			static_assert(false, "unsupported platform");
#endif
		} else {
			m_file_descriptor = -1;
		}

		{
			std::error_code ec;
			if (fs::is_directory(filepath, ec) && !ec) {
				// FIXME: can this error message be provided by the standard library?
				auto *msg = strerror(EISDIR);
				// FIXME: should be OSError
				return Err(value_error("{}", msg));
			}
		}

		if (auto dict = PyDict::create(); dict.is_ok()) {
			m_attributes = dict.unwrap();
		} else {
			return Err(dict.unwrap_err());
		}

		if (auto err = setattribute(PyString::create("name").unwrap(), filename); err.is_err()) {
			return Err(err.unwrap_err());
		}

		return Ok(0);
	}

	PyResult<std::ios_base::openmode> get_mode(const std::bitset<8> &rawmode)
	{
		bool rwa = false;
		bool plus = false;
		int mode{ 0 };

		if (rawmode.test(Mode::CREATE)) {
			if (rwa) {
				return Err(
					value_error("Must have exactly one of create/read/write/append "
								"mode and at most one plus"));
			}
			rwa = true;
			m_created = true;
			m_writable = true;
			mode |= std::ios_base::out;
		}
		if (rawmode.test(Mode::READ)) {
			if (rwa) {
				return Err(
					value_error("Must have exactly one of create/read/write/append "
								"mode and at most one plus"));
			}
			rwa = true;
			m_readable = true;
			mode |= std::ios_base::in;
		}

		if (rawmode.test(Mode::WRITE)) {
			if (rwa) {
				return Err(
					value_error("Must have exactly one of create/read/write/append "
								"mode and at most one plus"));
			}
			rwa = true;
			m_writable = true;
			mode |= std::ios_base::out;
		}

		if (rawmode.test(Mode::APPEND)) {
			if (rwa) {
				return Err(
					value_error("Must have exactly one of create/read/write/append "
								"mode and at most one plus"));
			}
			rwa = true;
			m_writable = true;
			m_appending = true;
			mode |= std::ios_base::out;
			mode |= std::ios_base::app;
		}

		if (rawmode.test(Mode::BINARY)) { mode |= std::ios_base::binary; }

		if (rawmode.test(Mode::UPDATE)) {
			if (plus) {
				return Err(
					value_error("Must have exactly one of create/read/write/append "
								"mode and at most one plus"));
			}
			plus = true;
			m_readable = true;
			m_writable = true;
			mode |= std::ios_base::in;
			mode |= std::ios_base::out;
		}

		if (!rwa) {
			return Err(
				value_error("Must have exactly one of create/read/write/append "
							"mode and at most one plus"));
		}

		return Ok(static_cast<std::ios_base::openmode>(mode));
	}

	std::string_view mode_string() const
	{
		if (m_created) {
			if (m_readable)
				return "xb+";
			else
				return "xb";
		}
		if (m_appending) {
			if (m_readable)
				return "ab+";
			else
				return "ab";
		} else if (m_readable) {
			if (m_writable)
				return "rb+";
			else
				return "rb";
		} else
			return "wb";
	}
};

class TextIOBase : public IOBase
{
	friend class ::Heap;

	TextIOBase() : TextIOBase(s_io_textiobase) {}

  protected:
	TextIOBase(PyType *type) : IOBase(type) {}

  public:
	static constexpr std::string_view __doc__ =
		"Base class for text I/O.\n"
		"\n"
		"This class provides a character and line based interface to stream\n"
		"I/O. There is no readinto method because Python's character strings\n"
		"are immutable. There is no public constructor.\n";

	static PyResult<TextIOBase *> create(const PyType *type)
	{
		auto &heap = VirtualMachine::the().heap();
		auto *result = heap.allocate<TextIOBase>(const_cast<PyType *>(type));
		if (!result) { return Err(memory_error(sizeof(TextIOBase))); }
		return Ok(result);
	}

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *, PyDict *)
	{
		return TextIOBase::create(type);
	}

	PyResult<PyObject *> detach() const
	{
		return Err(unsupported_operation(PyTuple::create(String{ "detach" }).unwrap(), nullptr));
	}

	PyResult<PyObject *> read(PyTuple *, PyDict *) const
	{
		return Err(unsupported_operation(PyTuple::create(String{ "read" }).unwrap(), nullptr));
	}

	PyResult<PyObject *> readline(PyTuple *, PyDict *) const
	{
		return Err(unsupported_operation(PyTuple::create(String{ "readline" }).unwrap(), nullptr));
	}

	PyResult<PyObject *> write(PyTuple *, PyDict *) const
	{
		return Err(unsupported_operation(PyTuple::create(String{ "write" }).unwrap(), nullptr));
	}

	PyType *static_type() const override { return s_io_textiobase; }

	static PyType *register_type(PyModule *module)
	{
		if (!s_io_textiobase) {
			s_io_textiobase =
				klass<TextIOBase>(module, "_TextIOBase", s_io_base)
					.def("detach", &TextIOBase::detach)
					.def("read", &TextIOBase::read)
					.def("readline", &TextIOBase::readline)
					.def("write", &TextIOBase::write)
					.property_readonly("encoding", [](TextIOBase *) { return Ok(py_none()); })
					.property_readonly("newlines", [](TextIOBase *) { return Ok(py_none()); })
					.property_readonly("errors", [](TextIOBase *) { return Ok(py_none()); })
					.finalize();
		}
		module->add_symbol(PyString::create("_TextIOBase").unwrap(), s_io_textiobase);
		return s_io_textiobase;
	}
};

class IncrementalNewlineDecoder : public PyBaseObject
{
	friend class ::Heap;
	PyObject *m_decoder{ nullptr };
	PyObject *m_errors{ nullptr };
	bool m_translate{ false };

	IncrementalNewlineDecoder() : IncrementalNewlineDecoder(s_io_incremental_newline_decoder) {}

  protected:
	IncrementalNewlineDecoder(PyType *type) : PyBaseObject(type) {}

  public:
	static constexpr std::string_view __doc__ =
		"IncrementalNewlineDecoder(decoder, translate, errors=\'strict\')\n"
		"--\n"
		"\n"
		"Codec used when reading a file in universal newlines mode.\n"
		"\n"
		"It wraps another incremental decoder, translating \\r\\n and \\r into \\n.\n"
		"It also records the types of newlines encountered.  When used with\n"
		"translate=False, it ensures that the newline sequence is returned in\n"
		"one piece. When used with decoder=None, it expects unicode strings as\n"
		"decode input and translates newlines without first invoking an external\n"
		"decoder.";

	static PyResult<IncrementalNewlineDecoder *> create(const PyType *type)
	{
		auto &heap = VirtualMachine::the().heap();
		auto *result = heap.allocate<IncrementalNewlineDecoder>(const_cast<PyType *>(type));
		if (!result) { return Err(memory_error(sizeof(IncrementalNewlineDecoder))); }
		return Ok(result);
	}

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *, PyDict *)
	{
		return IncrementalNewlineDecoder::create(type);
	}

	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs)
	{
		(void)args;
		(void)kwargs;
		TODO();
	}

	static PyType *register_type(PyModule *module)
	{
		if (!s_io_incremental_newline_decoder) {
			s_io_incremental_newline_decoder =
				klass<IncrementalNewlineDecoder>(module, "IncrementalNewlineDecoder").finalize();
		}
		module->add_symbol(PyString::create("IncrementalNewlineDecoder").unwrap(),
			s_io_incremental_newline_decoder);
		return s_io_incremental_newline_decoder;
	}

	void visit_graph(Visitor &visitor) override
	{
		PyObject::visit_graph(visitor);
		if (m_decoder) visitor.visit(*m_decoder);
		if (m_errors) visitor.visit(*m_errors);
	}
};

class StringIO : public TextIOBase
{
	friend ::Heap;

	std::stringstream m_stringstream;
	std::string m_newline;
	bool m_closed{ false };
	bool m_ok{ false };

	StringIO(PyType *type) : TextIOBase(type) {}

  public:
	static constexpr std::string_view __doc__ =
		"StringIO(initial_value=\'\', newline=\'\\n\')\n"
		"--\n"
		"\n"
		"Text I/O implementation using an in-memory buffer.\n"
		"\n"
		"The initial_value argument sets the value of object.  The newline\n"
		"argument is like the one of TextIOWrapper\'s constructor.";

	static PyResult<StringIO *> create(PyString *initial_value, PyString *newline)
	{
		return StringIO::__new__(s_io_stringio, nullptr, nullptr)
			.and_then([initial_value, newline](PyObject *stringio) -> PyResult<StringIO *> {
				auto result = static_cast<StringIO *>(stringio)->init(initial_value, newline);
				if (result.is_err()) return Err(result.unwrap_err());
				if (result.unwrap() != 0) { TODO(); }
				return Ok(static_cast<StringIO *>(stringio));
			});
	}

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *, PyDict *)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<StringIO>(const_cast<PyType *>(type))) { return Ok(obj); }
		return Err(memory_error(sizeof(StringIO)));
	}

	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs)
	{
		ASSERT(!kwargs || kwargs->map().empty());

		if (args->size() > 2) {
			return Err(type_error("StringIO() takes at most 2 arguments ({} given)", args->size()));
		}

		auto *initial_value = [args]() -> PyObject * {
			if (args->size() > 0) {
				return PyObject::from(args->elements()[0]).unwrap();
			} else {
				return PyString::create("").unwrap();
			}
		}();

		auto *newline = [args]() -> PyObject * {
			if (args->size() > 1) {
				return PyObject::from(args->elements()[1]).unwrap();
			} else {
				return PyString::create("\n").unwrap();
			}
		}();

		if (initial_value == py_none()) {
			initial_value = PyString::create("").unwrap();
		} else if (!as<PyString>(initial_value)) {
			return Err(
				type_error("newline must be str or None, not '{}'", initial_value->type()->name()));
		}

		if (newline == py_none()) {
			newline = PyString::create("\n").unwrap();
		} else if (!as<PyString>(newline)) {
			return Err(
				type_error("newline must be str or None, not '{}'", newline->type()->name()));
		}

		return init(as<PyString>(initial_value), as<PyString>(newline));
	}

	PyType *static_type() const override { return s_io_stringio; }

	PyResult<PyObject *> readable() const
	{
		return check_initialized()
			.and_then([this](auto) { return check_closed(); })
			.and_then([](auto) { return Ok(py_true()); });
	}

	PyResult<PyObject *> writable() const
	{
		return check_initialized()
			.and_then([this](auto) { return check_closed(); })
			.and_then([](auto) { return Ok(py_true()); });
	}

	PyResult<PyObject *> seekable() const
	{
		return check_initialized()
			.and_then([this](auto) { return check_closed(); })
			.and_then([](auto) { return Ok(py_true()); });
	}

	PyResult<PyObject *> closed() const
	{
		return check_initialized().and_then(
			[this](auto) { return Ok(m_closed ? py_true() : py_false()); });
	}

	PyResult<PyObject *> line_buffering() const
	{
		return check_initialized()
			.and_then([this](auto) { return check_closed(); })
			.and_then([](auto) { return Ok(py_false()); });
	}

	PyResult<PyObject *> newlines() const
	{
		return check_initialized()
			.and_then([this](auto) { return check_closed(); })
			.and_then([](auto) { return Ok(py_none()); });
	}

	PyResult<PyObject *> close()
	{
		m_closed = true;
		m_stringstream.clear();
		return Ok(py_none());
	}

	PyResult<PyObject *> tell()
	{
		return check_initialized()
			.and_then([this](auto) { return check_closed(); })
			.and_then([this](auto) { return PyInteger::create(m_stringstream.tellg()); });
	}

	PyResult<PyObject *> read(PyTuple *args, PyDict *kwargs)
	{
		ASSERT(!kwargs || kwargs->size() == 0);

		if (auto result = check_initialized().or_else([this](auto) { return check_closed(); });
			result.is_err()) {
			return result;
		}

		auto size_ = [args, this]() -> PyResult<size_t> {
			const auto initial_pos = m_stringstream.tellg();
			m_stringstream.seekg(0, std::ios_base::end);
			ASSERT(m_stringstream.tellg() >= initial_pos);
			const size_t str_size = m_stringstream.tellg() - initial_pos;
			m_stringstream.seekg(initial_pos);
			if (args->size() == 0) {
				return Ok(str_size);
			} else if (args->size() == 1) {
				auto obj = PyObject::from(args->elements()[0]);
				if (obj.is_err()) return Err(obj.unwrap_err());
				if (!as<PyInteger>(obj.unwrap())) {
					return Err(type_error("argument should be integer or None, not '{}'",
						obj.unwrap()->type()->to_string()));
				}
				auto size = as<PyInteger>(obj.unwrap())->as_size_t();
				return Ok(std::clamp(size, size_t{ 0 }, str_size));
			} else {
				return Err(type_error("read expected at most 1 argument, got {}", args->size()));
			}
		}();

		if (size_.is_err()) return Err(size_.unwrap_err());
		auto size = size_.unwrap();

		auto *result = new char[size];
		m_stringstream.read(result, size);
		return PyString::create(std::string{ result, size });
	}

	PyResult<PyObject *> readline(PyTuple *args, PyDict *kwargs)
	{
		ASSERT(!kwargs || kwargs->size() == 0);

		if (auto result = check_initialized().or_else([this](auto) { return check_closed(); });
			result.is_err()) {
			return result;
		}

		auto limit_ = [args, this]() -> PyResult<size_t> {
			const auto initial_pos = m_stringstream.tellg();
			m_stringstream.seekg(0, std::ios_base::end);
			ASSERT(m_stringstream.tellg() >= initial_pos);
			const size_t str_size = m_stringstream.tellg() - initial_pos;
			m_stringstream.seekg(initial_pos);
			if (args->size() == 0) {
				return Ok(str_size);
			} else if (args->size() == 1) {
				auto obj = PyObject::from(args->elements()[0]);
				if (obj.is_err()) return Err(obj.unwrap_err());
				if (!as<PyInteger>(obj.unwrap())) {
					return Err(type_error("argument should be integer or None, not '{}'",
						obj.unwrap()->type()->to_string()));
				}
				auto size = as<PyInteger>(obj.unwrap())->as_size_t();
				return Ok(std::clamp(size, size_t{ 0 }, str_size));
			} else {
				return Err(type_error("read expected at most 1 argument, got {}", args->size()));
			}
		}();

		if (limit_.is_err()) return Err(limit_.unwrap_err());
		auto limit = limit_.unwrap();

		if (m_newline == "\r\n") {
			TODO();
		} else if (m_newline.size() == 1) {
			// +1 for null terminator. I think?
			char *result = new char[limit + 1];
			m_stringstream.getline(result, limit + 1, m_newline.back());
			return PyString::create(std::string{ result, limit });
		} else {
			// +1 for null terminator. I think?
			char *result = new char[limit + 1];
			m_stringstream.getline(result, limit + 1);
			return PyString::create(std::string{ result, limit });
		}
	}

	PyResult<PyObject *> write(PyTuple *args, PyDict *kwargs)
	{
		if (auto result = check_initialized(); result.is_err()) { return result; }

		auto parse_result = PyArgsParser<PyString *>::unpack_tuple(args,
			kwargs,
			"write",
			std::integral_constant<size_t, 1>{},
			std::integral_constant<size_t, 1>{});

		if (parse_result.is_err()) return Err(parse_result.unwrap_err());

		auto [obj] = parse_result.unwrap();

		if (auto result = check_closed(); result.is_err()) { return result; }

		const auto size = obj->size();
		m_stringstream << obj->value();
		m_stringstream.seekg(m_stringstream.tellp());

		return PyInteger::create(size);
	}

	PyResult<PyObject *> seek(PyTuple *args, PyDict *kwargs)
	{
		ASSERT(!kwargs || kwargs->size() == 0);

		if (auto result = check_initialized(); result.is_err()) { return result; }

		auto parse_result = PyArgsParser<PyInteger *, PyInteger *>::unpack_tuple(args,
			kwargs,
			"seek",
			std::integral_constant<size_t, 1>{},
			std::integral_constant<size_t, 2>{},
			PyInteger::create(0).unwrap());

		if (parse_result.is_err()) return Err(parse_result.unwrap_err());

		auto [pos_, whence] = parse_result.unwrap();

		if (whence->as_i64() < 0 || whence->as_i64() > 2) {
			return Err(value_error("Invalid whence ('{}', should be 0, 1 or 2)", whence->as_i64()));
		} else if (pos_->as_i64() < 0 && whence->as_i64() == 0) {
			return Err(value_error("Negative seek position '{}'", pos_->as_i64()));
		} else if (whence->as_i64() != 0 && pos_->as_i64() == 0) {
			return Err(value_error("Can't do nonzero cur-relative seeks"));
		}

		if (whence->as_size_t() == 0) {
			const auto pos = pos_->as_i64();
			m_stringstream.seekp(pos, std::ios_base::beg);
			m_stringstream.seekg(pos, std::ios_base::beg);
		} else if (whence->as_size_t() == 1) {
		} else if (whence->as_size_t() == 2) {
			m_stringstream.seekp(0, std::ios_base::end);
			m_stringstream.seekg(0, std::ios_base::end);
		}

		const auto new_pos = m_stringstream.tellg();
		return PyInteger::create(new_pos);
	}

	PyResult<PyObject *> getvalue() { return PyString::create(m_stringstream.str()); }

	static PyType *register_type(PyModule *module)
	{
		if (!s_io_stringio) {
			s_io_stringio =
				klass<StringIO>(module, "StringIO", s_io_textiobase)
					.def("close", &StringIO::close)
					.def("getvalue", &StringIO::getvalue)
					.def("read", &StringIO::read)
					.def("readline", &StringIO::readline)
					.def("tell", &StringIO::tell)
					// .def("truncate", &StringIO::truncate)
					.def("seek", &StringIO::seek)
					.def("write", &StringIO::write)
					.def("seekable", &StringIO::seekable)
					.def("readable", &StringIO::readable)
					.def("writable", &StringIO::writable)
					.property_readonly("closed", [](StringIO *self) { return self->closed(); })
					.property_readonly("newlines", [](StringIO *self) { return self->newlines(); })
					.property_readonly(
						"line_buffering", [](StringIO *self) { return self->line_buffering(); })
					.finalize();
		}
		module->add_symbol(PyString::create("StringIO").unwrap(), s_io_stringio);
		return s_io_stringio;
	}


  private:
	PyResult<int32_t> init(PyString *initial_value, PyString *newline)
	{
		(void)initial_value;
		if (!newline->value().empty() && newline->value() != "\n" && newline->value() != "\r"
			&& newline->value() != "\r\n") {
			return Err(value_error("illegal newline value: '{}'", newline->to_string()));
		}
		m_newline = newline->value();
		const auto initial_str = initial_value->to_string();
		m_stringstream << initial_str;
		m_closed = false;
		m_ok = true;

		return Ok(0);
	}

	PyResult<PyObject *> check_initialized() const
	{
		if (!m_ok) { return Err(value_error("I/O operation on uninitialized object")); }
		return Ok(py_true());
	}

	PyResult<PyObject *> check_closed() const
	{
		if (m_closed) { return Err(value_error("I/O operation on closed file")); }
		return Ok(py_true());
	}
};


class TextIOWrapper : public TextIOBase
{
	friend ::Heap;

	PyObject *m_buffer{ nullptr };
	std::string m_errors;
	std::string m_encoding;
	std::string m_newline;
	bool m_line_buffering;
	bool m_write_through;

	TextIOWrapper(PyType *type) : TextIOBase(type) {}

  public:
	static constexpr std::string_view __doc__ =
		"TextIOWrapper(buffer, encoding=None, errors=None, newline=None,\n"
		"              line_buffering=False, write_through=False)\n"
		"--\n"
		"\n"
		"Character and line based layer over a BufferedIOBase object, buffer.\n"
		"\n"
		"encoding gives the name of the encoding that the stream will be\n"
		"decoded or encoded with. It defaults to locale.getpreferredencoding(False).\n"
		"\n"
		"errors determines the strictness of encoding and decoding (see\n"
		"help(codecs.Codec) or the documentation for codecs.register) and\n"
		"defaults to \"strict\".\n"
		"\n"
		"newline controls how line endings are handled. It can be None, \'\',\n"
		"\'\\n\', \'\\r\', and \'\\r\\n\'.  It works as follows:\n"
		"\n"
		"* On input, if newline is None, universal newlines mode is\n"
		"  enabled. Lines in the input can end in \'\\n\', \'\\r\', or \'\\r\\n\', and\n"
		"  these are translated into \'\\n\' before being returned to the\n"
		"  caller. If it is \'\', universal newline mode is enabled, but line\n"
		"  endings are returned to the caller untranslated. If it has any of\n"
		"  the other legal values, input lines are only terminated by the given\n"
		"  string, and the line ending is returned to the caller untranslated.\n"
		"\n"
		"* On output, if newline is None, any \'\\n\' characters written are\n"
		"  translated to the system default line separator, os.linesep. If\n"
		"  newline is \'\' or \'\\n\', no translation takes place. If newline is any\n"
		"  of the other legal values, any \'\\n\' characters written are translated\n"
		"  to the given string.\n"
		"\n"
		"If line_buffering is True, a call to flush is implied when a call to\n"
		"write contains a newline character.";

	static PyResult<TextIOWrapper *> create()
	{
		return StringIO::__new__(s_io_stringio, nullptr, nullptr).and_then([](auto *obj) {
			return Ok(static_cast<TextIOWrapper *>(obj));
		});
	}

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *, PyDict *)
	{
		auto &heap = VirtualMachine::the().heap();
		if (auto *obj = heap.allocate<TextIOWrapper>(const_cast<PyType *>(type))) {
			return Ok(obj);
		}
		return Err(memory_error(sizeof(TextIOWrapper)));
	}

	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs)
	{
		auto parse_result =
			PyArgsParser<PyObject *, PyObject *, PyObject *, PyObject *, PyBool *, PyBool *>::
				unpack_tuple(args,
					kwargs,
					"TextIOWrapper",
					std::integral_constant<size_t, 1>{},
					std::integral_constant<size_t, 6>{},
					py_none(),
					py_none(),
					py_none(),
					static_cast<PyBool *>(py_false()),
					static_cast<PyBool *>(py_false()));

		if (parse_result.is_err()) return Err(parse_result.unwrap_err());
		auto buffer = std::get<0>(parse_result.unwrap());
		auto encoding_ = std::get<1>(parse_result.unwrap());
		auto errors_ = std::get<2>(parse_result.unwrap());
		auto newline_ = std::get<3>(parse_result.unwrap());
		auto line_buffering = std::get<4>(parse_result.unwrap());
		auto write_through = std::get<5>(parse_result.unwrap());

		auto encoding = [&]() -> PyResult<std::optional<std::string>> {
			if (encoding_ == py_none()) { return Ok(std::nullopt); }
			if (!as<PyString>(encoding_)) {
				return Err(
					type_error("TextIOWrapper() argument 'encoding' must be str or None, not '{}'",
						encoding_->type()->name()));
			} else {
				return Ok(as<PyString>(encoding_)->value());
			}
		}();
		if (encoding.is_err()) { return Err(encoding.unwrap_err()); }

		auto newline = [&]() -> PyResult<std::optional<std::string>> {
			if (newline_ == py_none()) { return Ok(std::nullopt); }
			if (!as<PyString>(newline_)) {
				return Err(
					type_error("TextIOWrapper() argument 'newline' must be str or None, not '{}'",
						newline_->type()->name()));
			} else {
				return Ok(as<PyString>(newline_)->value());
			}
		}();
		if (newline.is_err()) { return Err(newline.unwrap_err()); }

		auto errors = [&]() -> PyResult<std::optional<std::string>> {
			if (errors_ == py_none()) { return Ok(std::nullopt); }
			if (!as<PyString>(errors_)) {
				return Err(
					type_error("TextIOWrapper() argument 'errors' must be str or None, not '{}'",
						errors_->type()->name()));
			} else {
				return Ok(as<PyString>(errors_)->value());
			}
		}();
		if (errors.is_err()) { return Err(errors.unwrap_err()); }

		return init(buffer,
			encoding.unwrap(),
			errors.unwrap(),
			newline.unwrap(),
			line_buffering == py_true(),
			write_through == py_true());
	}

	PyType *static_type() const override { return s_io_textiowrapper; }

	static PyType *register_type(PyModule *module)
	{
		if (!s_io_textiowrapper) {
			s_io_textiowrapper =
				klass<TextIOWrapper>(module, "TextIOWrapper", s_io_textiobase).finalize();
		}
		module->add_symbol(PyString::create("StringIO").unwrap(), s_io_textiowrapper);
		return s_io_textiowrapper;
	}

	void visit_graph(Visitor &visitor) override
	{
		PyObject::visit_graph(visitor);
		if (m_buffer) visitor.visit(*m_buffer);
	}

  private:
	PyResult<int32_t> init(PyObject *buffer,
		const std::optional<std::string> &encoding,
		const std::optional<std::string> &errors,
		const std::optional<std::string> &newline,
		bool line_buffering,
		bool write_through)
	{
		(void)buffer;
		(void)encoding;
		(void)errors;
		(void)newline;
		(void)line_buffering;
		(void)write_through;
		TODO();
	}
};

PyResult<PyObject *> open(PyObject *file, const std::string &mode)
{
	const auto flag_ = read_flags(mode);
	if (flag_.is_err()) return Err(flag_.unwrap_err());
	const auto flag = flag_.unwrap();

	auto raw = FileIO::create(file, flag, false, py_none());
	if (raw.is_err()) return raw;

	int buffering = -1;

	auto buffer = [&flag, &raw, buffering, &mode]() -> PyResult<PyObject *> {
		if (flag.test(Mode::UPDATE)) {
			TODO();
		} else if (flag.test(Mode::CREATE) || flag.test(Mode::WRITE) || flag.test(Mode::APPEND)) {
			TODO();
		} else if (flag.test(Mode::READ)) {
			return BufferedReader::create(raw.unwrap(), buffering);
		} else {
			return Err(value_error("unknown mode: '{}'", mode));
		}
	}();

	if (buffer.is_err()) return buffer;

	if (flag.test(Mode::BINARY)) { return buffer; }

	TODO();
}

// TODO: move this to a header file since it is part of builtin module
class BlockingIOError : public OSError
{
	friend class ::Heap;

  private:
	BlockingIOError(PyType *t) : OSError(t) {}

	BlockingIOError(PyTuple *args) : OSError(s_blocking_io_error, args) {}

	static BlockingIOError *create(PyTuple *args)
	{
		auto &heap = VirtualMachine::the().heap();
		return heap.allocate<BlockingIOError>(args);
	}

  public:
	static constexpr std::string_view __doc__ = "I/O operation would block.";

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs)
	{
		ASSERT(type == s_blocking_io_error);
		ASSERT(!kwargs || kwargs->map().empty());
		return Ok(BlockingIOError::create(args));
	}

	static PyType *register_type(PyModule *module)
	{
		if (!s_blocking_io_error) {
			s_blocking_io_error =
				klass<BlockingIOError>(module, "BlockingIOError", OSError::class_type())
					.finalize();
		} else {
			module->add_symbol(PyString::create("BlockingIOError").unwrap(), s_blocking_io_error);
		}
		return s_blocking_io_error;
	}

	PyType *static_type() const override
	{
		ASSERT(s_blocking_io_error);
		return s_blocking_io_error;
	}
};

PyModule *io_module()
{
	auto *s_io_module = PyModule::create(PyDict::create().unwrap(),
		PyString::create("_io").unwrap(),
		PyString::create("The _io module!").unwrap())
							.unwrap();

	(void)IOBase::register_type(s_io_module);
	(void)RawIOBase::register_type(s_io_module);
	(void)BufferedIOBase::register_type(s_io_module);
	(void)TextIOBase::register_type(s_io_module);
	(void)BufferedReader::register_type(s_io_module);
	(void)BufferedWriter::register_type(s_io_module);
	(void)BufferedRWPair::register_type(s_io_module);
	(void)BufferedRandom::register_type(s_io_module);
	(void)IncrementalNewlineDecoder::register_type(s_io_module);
	(void)BytesIO::register_type(s_io_module);
	(void)FileIO::register_type(s_io_module);
	(void)StringIO::register_type(s_io_module);
	(void)TextIOWrapper::register_type(s_io_module);

	(void)BlockingIOError::register_type(s_io_module);

	s_io_module->add_symbol(PyString::create("open").unwrap(),
		VirtualMachine::the().heap().allocate<PyNativeFunction>(
			"open", [](PyTuple *args, PyDict *kwargs) {
				ASSERT(!kwargs || kwargs->map().empty());
				ASSERT(args && args->elements().size() == 2);
				auto arg0 = PyObject::from(args->elements()[0]).unwrap();
				auto arg1 = PyObject::from(args->elements()[1]).unwrap();

				ASSERT(as<PyString>(arg1));
				const std::string rawmode = as<PyString>(arg1)->value();

				return open(arg0, rawmode);
			}));

	s_io_module->add_symbol(PyString::create("open_code").unwrap(),
		VirtualMachine::the().heap().allocate<PyNativeFunction>(
			"open", [](PyTuple *args, PyDict *kwargs) {
				ASSERT(!kwargs || kwargs->map().empty());
				ASSERT(args && args->elements().size() == 1);
				auto arg0 = PyObject::from(args->elements()[0]).unwrap();

				return open(arg0, "rb");
			}));

	// C++ standard streams currently do not provide an API to get default buffer size, and C's
	// BUFSIZE doesn't have to be respected
	s_io_module->add_symbol(PyString::create("DEFAULT_BUFFER_SIZE").unwrap(), Number{ 0 });

	// >> type("UnsupportedOperation", (_io.OSError, ValueError), {})
	auto unsupported_operation_type = type()->call(
		PyTuple::create(PyString::create("UnsupportedOperation").unwrap(),
			PyTuple::create(OSError::class_type(), ValueError::class_type()).unwrap(),
			PyDict::create().unwrap())
			.unwrap(),
		PyDict::create().unwrap());

	ASSERT(unsupported_operation_type.is_ok());
	ASSERT(as<PyType>(unsupported_operation_type.unwrap()));

	s_unsupported_operation_type = as<PyType>(unsupported_operation_type.unwrap());

	s_io_module->add_symbol(
		PyString::create("UnsupportedOperation").unwrap(), s_unsupported_operation_type);

	return s_io_module;
}
}// namespace py
