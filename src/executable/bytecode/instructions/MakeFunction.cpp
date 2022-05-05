#include "MakeFunction.hpp"
#include "executable/Mangler.hpp"
#include "runtime/PyTuple.hpp"


using namespace py;

PyResult<Value> MakeFunction::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	std::vector<Value> default_values;
	default_values.reserve(m_defaults_size);
	if (m_defaults_size > 0) {
		auto *end = vm.stack_pointer() + m_defaults_stack_offset + m_defaults_size;
		for (auto *sp = vm.stack_pointer() + m_defaults_stack_offset; sp < end; ++sp) {
			default_values.push_back(*sp);
		}
	}

	std::vector<Value> kw_default_values;
	kw_default_values.reserve(m_kw_defaults_size);
	if (m_kw_defaults_size > 0) {
		auto *end = vm.stack_pointer() + m_kw_defaults_stack_offset + m_kw_defaults_size;
		for (auto *sp = vm.stack_pointer() + m_kw_defaults_stack_offset; sp < end; ++sp) {
			kw_default_values.push_back(*sp);
		}
	}

	auto closure = [&]() -> std::variant<std::vector<PyCell *>, BaseException *> {
		if (m_captures_tuple) {
			auto value_ = PyObject::from(vm.reg(*m_captures_tuple));
			if (value_.is_err()) { return value_.unwrap_err(); }
			auto *value = value_.unwrap();
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
			return std::vector<PyCell *>{};
		}
	}();

	if (std::holds_alternative<BaseException *>(closure)) {
		return Err(std::get<BaseException *>(closure));
	}

	const auto &function_name_value = vm.reg(m_name);
	ASSERT(std::holds_alternative<String>(function_name_value));
	const auto function_name = std::get<String>(function_name_value).s;

	auto *func = interpreter.make_function(
		function_name, default_values, kw_default_values, std::get<std::vector<PyCell *>>(closure));
	ASSERT(func)
	// const std::string demangled_name =
	// Mangler::default_mangler().function_demangle(function_name);
	vm.reg(m_dst) = func;
	return Ok(Value{ func });
}

std::vector<uint8_t> MakeFunction::serialize() const
{
	ASSERT(m_defaults_size < std::numeric_limits<uint8_t>::max());
	ASSERT(m_defaults_stack_offset < std::numeric_limits<uint8_t>::max());
	ASSERT(m_kw_defaults_size < std::numeric_limits<uint8_t>::max());
	ASSERT(m_kw_defaults_stack_offset < std::numeric_limits<uint8_t>::max());

	return {
		MAKE_FUNCTION,
		m_dst,
		m_name,
		static_cast<uint8_t>(m_defaults_size),
		static_cast<uint8_t>(m_defaults_stack_offset),
		static_cast<uint8_t>(m_kw_defaults_size),
		static_cast<uint8_t>(m_kw_defaults_stack_offset),
		static_cast<uint8_t>(m_captures_tuple.has_value()),
		m_captures_tuple.value_or(0),
	};
}