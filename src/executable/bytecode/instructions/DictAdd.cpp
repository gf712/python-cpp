module;
#include "core.hpp"
#include "executable/Label.hpp"
#include <cstddef>
#include <cstdint>

module py.runtime;
import std;

// After the import: these name module-owned types.


using namespace py;

PyResult<Value> DictAdd::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &dict = vm.reg(m_dict);
	auto &key = vm.reg(m_key);
	auto &value = vm.reg(m_value);

	ASSERT(std::holds_alternative<PyObject *>(dict));

	auto *pydict = std::get<PyObject *>(dict);
	ASSERT(pydict);
	ASSERT(as<PyDict>(pydict));

	as<PyDict>(pydict)->insert(key, value);

	return Ok(py_none());
}

std::vector<uint8_t> DictAdd::serialize() const
{
	return {
		DICT_ADD,
		m_dict,
		m_key,
		m_value,
	};
}
