#include "DictMerge.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyTuple.hpp"

using namespace py;

void DictMerge::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &this_dict = vm.reg(m_this_dict);
	auto &other_dict = vm.reg(m_other_dict);

	ASSERT(std::holds_alternative<PyObject *>(this_dict))
	ASSERT(std::holds_alternative<PyObject *>(other_dict))

	auto *this_pydict = std::get<PyObject *>(this_dict);
	ASSERT(this_pydict)
	ASSERT(as<PyDict>(this_pydict))

	auto *other_pydict = std::get<PyObject *>(other_dict);
	ASSERT(other_pydict)
	ASSERT(as<PyDict>(other_pydict))

	as<PyDict>(this_pydict)->merge(PyTuple::create(other_pydict), nullptr);
}

std::vector<uint8_t> DictMerge::serialize() const
{
	return {
		MERGE_DICT,
		m_this_dict,
		m_other_dict,
	};
}
