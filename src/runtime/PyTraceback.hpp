#pragma once

#include "PyObject.hpp"

namespace py {

class PyTraceback : public PyBaseObject
{
  public:
	// frame object at this level
	PyFrame *m_tb_frame = nullptr;
	// index of last attempted instruction in bytecode
	size_t m_tb_lasti;
	// current line number in Python source code
	size_t m_tb_lineno;
	// next inner traceback object (called by this level)
	PyTraceback *m_tb_next = nullptr;

  private:
	friend class ::Heap;

	PyTraceback(PyFrame *tb_frame, size_t tb_lasti, size_t tb_lineno, PyTraceback *tb_next);

  public:
	static PyResult
		create(PyFrame *tb_frame, size_t tb_lasti, size_t tb_lineno, PyTraceback *tb_next);

	std::string to_string() const override;

	PyResult __repr__() const;

	void visit_graph(Visitor &visitor) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};

}// namespace py