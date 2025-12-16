#include "MakeFunction.hpp"
#include "executable/Mangler.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyCode.hpp"
#include "runtime/PyFrame.hpp"
#include "runtime/PyTuple.hpp"
#include "vm/VM.hpp"


using namespace py;

PyResult<Value> MakeFunction::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	std::vector<Value> default_values;
	default_values.reserve(m_defaults_size);
	auto *start = vm.sp() - m_defaults_size - m_kw_defaults_size;
	while (start != (vm.sp() - m_kw_defaults_size)) {
		default_values.push_back(*start);
		start = std::next(start);
	}

	std::vector<Value> kw_default_values;
	kw_default_values.reserve(m_kw_defaults_size);
	start = vm.sp() - m_kw_defaults_size;
	while (start != vm.sp()) {
		kw_default_values.push_back(*start);
		start = std::next(start);
	}

	auto closure = [&]() -> PyResult<PyTuple *> {
		if (m_captures_tuple) {
			auto cells = PyObject::from(vm.reg(*m_captures_tuple));
			if (cells.is_err()) return Err(cells.unwrap_err());
			ASSERT(as<PyTuple>(cells.unwrap()));
			return Ok(as<PyTuple>(cells.unwrap()));
		} else {
			return Ok(nullptr);
		}
	}();

	if (closure.is_err()) { return Err(closure.unwrap_err()); }

	const auto &function_name_value = vm.reg(m_name);
	ASSERT(std::holds_alternative<String>(function_name_value));
	const auto function_name = std::get<String>(function_name_value).s;

	auto *func = interpreter.execution_frame()->code()->make_function(
		function_name, default_values, kw_default_values, closure.unwrap());
	ASSERT(func);
	// const std::string demangled_name =
	// Mangler::default_mangler().function_demangle(function_name);
	vm.reg(m_dst) = func;
	return Ok(Value{ func });
}

std::vector<uint8_t> MakeFunction::serialize() const
{
	ASSERT(m_defaults_size < std::numeric_limits<uint8_t>::max());
	ASSERT(m_kw_defaults_size < std::numeric_limits<uint8_t>::max());

	return {
		MAKE_FUNCTION,
		m_dst,
		m_name,
		static_cast<uint8_t>(m_defaults_size),
		static_cast<uint8_t>(m_kw_defaults_size),
		static_cast<uint8_t>(m_captures_tuple.has_value()),
		m_captures_tuple.value_or(0),
	};
}