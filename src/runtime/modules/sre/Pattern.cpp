#include "Pattern.hpp"
#include "Match.hpp"
#include "memory/Heap.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyMappingProxy.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/ValueError.hpp"
#include "runtime/types/api.hpp"
#include "runtime/types/builtin.hpp"
#include "utilities.hpp"
#include "vm/VM.hpp"
#include <cstdint>
#include <limits>

using namespace py;
using namespace py::sre;

namespace {
static PyType *s_sre_pattern = nullptr;
}

Pattern::Pattern(PyType *type) : PyBaseObject(type) {}

Pattern::Pattern(size_t groups,
	PyDict *groupindex,
	PyTuple *indexgroup,
	int32_t flags,
	PyObject *pattern,
	std::optional<bool> isbytes,
	std::vector<uint32_t> code)
	: PyBaseObject(s_sre_pattern), m_groups(groups), m_groupindex(groupindex),
	  m_indexgroup(indexgroup), m_flags(flags), m_pattern(pattern), m_isbytes(isbytes), m_code(code)
{}

PyResult<Pattern *> Pattern::create(PyObject *pattern,
	int32_t flags,
	PyList *code,
	size_t groups,
	PyDict *groupindex,
	PyTuple *indexgroup)
{
	const auto codesize = code->elements().size();

	std::vector<uint32_t> code_vec;
	code_vec.reserve(codesize);
	for (const auto &el : code->elements()) {
		auto el_ = PyObject::from(el);
		if (el_.is_err()) { return Err(el_.unwrap_err()); }
		if (!el_.unwrap()->type()->issubclass(types::integer())) { TODO(); }
		auto value = static_cast<const PyInteger &>(*el_.unwrap()).as_size_t();
		if (!fits_in<uint32_t>(value)) {
			return Err(value_error("code value {} does not fit in [{}, {}]",
				value,
				std::numeric_limits<uint32_t>::min(),
				std::numeric_limits<uint32_t>::max()));
		}
		code_vec.push_back(static_cast<uint32_t>(value));
	}

	std::optional<bool> isbytes;
	if (pattern != py_none()) {
		if (pattern->type()->issubclass(types::str())) {
			isbytes = false;
		} else if (pattern->type()->issubclass(types::bytes())
				   || pattern->type()->issubclass(types::bytearray())) {
			isbytes = true;
		} else {
			return Err(type_error("expected string or bytes-like object"));
		}
	}

	auto obj = VirtualMachine::the().heap().allocate<Pattern>(
		groups, groupindex, indexgroup, flags, pattern, std::move(isbytes), std::move(code_vec));
	if (!obj) { return Err(memory_error(sizeof(Pattern))); }
	return Ok(obj);
}

PyResult<PyObject *> Pattern::match(PyTuple *args, PyDict *kwargs)
{
	(void)args;
	(void)kwargs;
	return Match::create();
}

void Pattern::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_groupindex) { visitor.visit(*m_groupindex); }
	if (m_indexgroup) { visitor.visit(*m_indexgroup); }
	if (m_pattern) { visitor.visit(*m_pattern); }
}

PyType *Pattern::register_type(PyModule *module)
{
	if (!s_sre_pattern) {
		s_sre_pattern = klass<Pattern>(module, "re.Pattern")
							.attribute_readonly("pattern", &Pattern::m_pattern)
							.attribute_readonly("flags", &Pattern::m_flags)
							.attribute_readonly("groups", &Pattern::m_groups)
							.property_readonly("groupindex",
								[](Pattern *pattern) -> PyResult<PyObject *> {
									if (!pattern->m_groupindex) { return PyDict::create(); }
									return PyMappingProxy::create(pattern->m_groupindex);
								})
							.def("match", &Pattern::match)
							.finalize();
	}
	return s_sre_pattern;
}