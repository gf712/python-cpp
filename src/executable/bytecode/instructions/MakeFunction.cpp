#include "MakeFunction.hpp"
#include "executable/Mangler.hpp"
#include "runtime/PyTuple.hpp"


using namespace py;

void MakeFunction::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	std::vector<Value> default_values;
	default_values.reserve(m_defaults.size());
	for (const auto &default_value : m_defaults) {
		default_values.push_back(vm.reg(default_value));
	}

	std::vector<Value> kw_default_values;
	kw_default_values.reserve(m_kw_defaults.size());
	for (const auto &default_value : m_kw_defaults) {
		if (default_value.has_value()) { kw_default_values.push_back(vm.reg(*default_value)); }
	}

	auto closure = [&]() -> std::vector<PyCell *> {
		if (m_captures_tuple) {
			auto *value = PyObject::from(vm.reg(*m_captures_tuple));
			ASSERT(as<PyTuple>(value))
			std::vector<PyCell *> cells;
			cells.reserve(as<PyTuple>(value)->size());
			for (const auto &el : as<PyTuple>(value)->elements()) {
				ASSERT(std::holds_alternative<PyObject *>(el))
				ASSERT(as<PyCell>(std::get<PyObject *>(el)))
				cells.push_back(as<PyCell>(std::get<PyObject *>(el)));
			}
			return cells;
		} else {
			return {};
		}
	}();

	auto *func =
		interpreter.make_function(m_function_name, default_values, kw_default_values, closure);
	ASSERT(func)
	// FIXME: demangle should be a function visible in the whole project
	const std::string demangled_name =
		Mangler::default_mangler().function_demangle(m_function_name);
	vm.reg(m_dst) = func;
}