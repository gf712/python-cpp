module;
#include "memory/allocate.hpp"

export module py.runtime:memory_view;
import :object;
import py.memory;
import std;

export namespace py {
struct Number;
class PyTuple;
class PyDict;
class PyType;

class PyMemoryView : public PyBaseObject
{
	friend class ::Heap;
	friend class py::detail::Allocator;

	struct ManagedBuffer
	{
		PyBuffer m_main_view;
	};

	std::shared_ptr<ManagedBuffer> m_managed_buffer;// the original view
	PyBuffer m_view;// our view

	PyMemoryView(PyType *);
	PyMemoryView(PyBuffer);

  public:
	static PyResult<PyObject *> create(PyObject *object);
	static PyResult<PyObject *> create(PyBuffer buffer);

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;

	PyResult<std::size_t> __len__() const;

	PyResult<PyObject *> cast(PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> tolist();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

	void visit_graph(Visitor &) override;
	std::string to_string() const override;

	std::size_t itemsize() const { return m_view.itemsize; }

	PyResult<std::monostate> __getbuffer__(PyBuffer &view, int /*flags*/);

  private:
	static PyResult<PyBuffer> create_view(PyBuffer &main_view);
};

}// namespace py
