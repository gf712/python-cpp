#pragma once

#include "runtime/PyDict.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/Value.hpp"
#include <cstdint>

namespace py {
namespace sre {
	class Pattern : public PyBaseObject
	{
		friend class ::Heap;

		size_t m_groups;
		PyDict *m_groupindex{ nullptr };
		PyTuple *m_indexgroup{ nullptr };
		int32_t m_flags;
		PyObject *m_pattern{ nullptr };
		std::optional<bool> m_isbytes;
		std::vector<uint32_t> m_code;

		Pattern(PyType *);

		Pattern(size_t groups,
			PyDict *groupindex,
			PyTuple *indexgroup,
			int32_t flags,
			PyObject *pattern,
			std::optional<bool> isbytes,
			std::vector<uint32_t> code);

	  public:
		static PyResult<Pattern *> create(PyObject *pattern,
			int32_t flags,
			PyList *code,
			size_t groups,
			PyDict *groupindex,
			PyTuple *indexgroup);

		PyResult<PyObject *> match(PyTuple *args, PyDict *kwargs);

		void visit_graph(Visitor &visitor) override;

		static PyType *register_type(PyModule *module);
	};
}// namespace sre
}// namespace py