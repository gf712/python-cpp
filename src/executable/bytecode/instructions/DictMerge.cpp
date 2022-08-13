#include "DictMerge.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyTuple.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> DictMerge::execute(VirtualMachine &vm, Interpreter &) const
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

	auto args = PyTuple::create(other_pydict);
	if (args.is_err()) { return Err(args.unwrap_err()); }
	if (auto result = as<PyDict>(this_pydict)->merge(args.unwrap(), nullptr); result.is_ok()) {
		return Ok(Value{ result.unwrap() });
	} else {
		return Err(result.unwrap_err());
	}
}

std::vector<uint8_t> DictMerge::serialize() const
{
	return {
		DICT_MERGE,
		m_this_dict,
		m_other_dict,
	};
}
