module;
#include "core.hpp"
#include "memory/allocate.hpp"

export module py.runtime:syntax_error;
import :baseexception;
import :dict;
import :exception;
import :object;
import :string;
import :tuple;
import :value;
import py.memory;
import std;

export namespace py {
class PyType;

// Where a syntax error happened, in CPython's (filename, lineno, offset, text)
// order -- the same order as the info tuple accepted by `SyntaxError(msg, info)`.
// `lineno` and `offset` are 1-based, and `offset` indexes into `text` (the source
// line), not into the file.
struct SyntaxErrorLocation
{
	std::string filename;
	std::size_t lineno;
	std::size_t offset;
	std::string text;
};

class SyntaxError : public Exception
{
	friend class ::Heap;
	friend class py::detail::Allocator;
	friend BaseException *syntax_error(std::string);
	friend BaseException *syntax_error(std::string, SyntaxErrorLocation);

  public:
	PyObject *m_msg{ nullptr };
	PyObject *m_filename{ nullptr };
	PyObject *m_text{ nullptr };
	PyObject *m_lineno{ nullptr };
	PyObject *m_offset{ nullptr };

  private:
	SyntaxError(PyType *type);
	SyntaxError(PyTuple *args);

	static SyntaxError *create(PyTuple *);

	static SyntaxError *create(std::string message);

	static SyntaxError *create(std::string message, SyntaxErrorLocation location);

  public:
	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	PyResult<std::int32_t> __init__(PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __str__() const;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;

	std::string format_exception_only() const override;

	void visit_graph(Visitor &) override;
};

inline BaseException *syntax_error(std::string message)
{
	return SyntaxError::create(std::move(message));
}

inline BaseException *syntax_error(std::string message, SyntaxErrorLocation location)
{
	return SyntaxError::create(std::move(message), std::move(location));
}

}// namespace py
