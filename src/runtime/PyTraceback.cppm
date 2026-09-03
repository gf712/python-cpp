module;

#include "memory/allocate.hpp"
#include <cstddef>

export module py.runtime:traceback;

import std;
import :object;
import py.memory;

export namespace py {
class PyType;


class PyFrame;

class PyTraceback : public PyBaseObject
{
  public:
	// frame object at this level
	PyFrame *m_tb_frame = nullptr;
	// index of last attempted instruction in bytecode
	std::size_t m_tb_lasti;
	// current line number in Python source code
	std::size_t m_tb_lineno;
	// next inner traceback object (called by this level)
	PyTraceback *m_tb_next = nullptr;

  private:
	friend class ::Heap;
	friend class py::detail::Allocator;

	PyTraceback(PyType *);

	PyTraceback(PyFrame *tb_frame,
		std::size_t tb_lasti,
		std::size_t tb_lineno,
		PyTraceback *tb_next);

  public:
	static PyResult<PyTraceback *> create(PyFrame *tb_frame,
		std::size_t tb_lasti,
		std::size_t tb_lineno,
		PyTraceback *tb_next);

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;

	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
