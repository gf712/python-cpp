#include "Match.hpp"
#include "memory/Heap.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyMappingProxy.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/types/api.hpp"
#include "runtime/types/builtin.hpp"
#include "utilities.hpp"
#include "vm/VM.hpp"

using namespace py;
using namespace py::sre;

namespace {
static PyType *s_sre_match = nullptr;
}

Match::Match(PyType *type) : PyBaseObject(type) {}

Match::Match() : PyBaseObject(s_sre_match) {}

PyResult<Match *> Match::create()
{
	auto obj = VirtualMachine::the().heap().allocate<Match>();
	if (!obj) { return Err(memory_error(sizeof(Match))); }
	return Ok(obj);
}

void Match::visit_graph(Visitor &visitor) { PyObject::visit_graph(visitor); }

PyType *Match::register_type(PyModule *module)
{
	if (!s_sre_match) { s_sre_match = klass<Match>(module, "_sre.Match").finalize(); }
	return s_sre_match;
}