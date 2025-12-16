#include "runtime/PyObject.hpp"
#include "runtime/Value.hpp"
#include <cstddef>

namespace py {
namespace itertools {
	class Permutations : public PyBaseObject
	{
		friend class ::Heap;

		PyList *m_pool{ nullptr };
		size_t m_length;
		size_t m_iterator_length;
		size_t m_inner_iteration;
		std::vector<size_t> m_indices;
		std::vector<size_t> m_cycles;
		bool m_done;
		bool m_first{ true };

		Permutations(PyType *);
		Permutations(PyList *pool, size_t length);
		static PyResult<PyObject *> create(PyObject *iterable, std::optional<size_t> length);

	  public:
		static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

		PyResult<PyObject *> __iter__() const;
		PyResult<PyObject *> __next__();

		static PyType *register_type(PyModule *module);

		void visit_graph(Visitor &visitor) override;
	};
}// namespace itertools
}// namespace py
