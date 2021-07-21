#include "Instructions.hpp"
#include "runtime/PyObject.hpp"

void StoreName::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &value = vm.reg(m_source);
	auto obj = std::visit(
		overloaded{ [](const Number &n) {
					   return std::static_pointer_cast<PyObject>(PyObjectNumber::create(n));
				   },
			[](const String &s) {
				return std::static_pointer_cast<PyObject>(PyString::create(s.s));
			},
			[](const NameConstant &s) {
				return PyObject::from(s);
			},
			[](const std::shared_ptr<PyObject> &obj) { return obj; },
			[&interpreter, this](const auto &) {
				interpreter.raise_exception("Failed to store object \"{}\"", m_object_name);
				return std::shared_ptr<PyObject>(nullptr);
			} },
		value);
	interpreter.store_object(m_object_name, obj);
}

void ReturnValue::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	vm.reg(0) = vm.reg(m_source);
	vm.set_instruction_pointer(vm.return_address());
	interpreter.set_execution_frame(interpreter.execution_frame()->parent());
}


void MakeFunction::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto code = interpreter.allocate_object<PyCode>(
		m_function_name + "__code__", vm.function_offset(m_function_id), m_args);
	interpreter.allocate_object<PyFunction>(
		m_function_name, std::static_pointer_cast<PyCode>(code));
}


void JumpIfFalse::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &result = vm.reg(m_test_register);

	const bool test_result =
		std::visit(overloaded{ [](const std::shared_ptr<PyObject> &obj) -> bool {
								  if (auto bool_obj = as<PyNameConstant>(obj)) {
									  const auto value = bool_obj->value();
									  if (auto *bool_type = std::get_if<bool>(&value.value)) {
										  return *bool_type;
									  } else {
										  return false;
									  }
								  }
								  TODO()
								  return false;
							  },
					   [](const auto &) -> bool {
						   TODO()
						   return false;
					   },
					   [](const NameConstant &value) -> bool {
						   if (auto *bool_type = std::get_if<bool>(&value.value)) {
							   return *bool_type;
						   } else {
							   return false;
						   }
					   } },
			result);

	if (!test_result) { vm.set_instruction_pointer(m_label.position()); }
};


void JumpIfFalse::rellocate(BytecodeGenerator &generator, const std::vector<size_t> &offsets)
{
	m_label = generator.label(m_label);
	const size_t offset = offsets[m_label.function_id()];
	m_label.set_position(m_label.position() + offset);
}

void Jump::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.set_instruction_pointer(m_label.position());
};


void Jump::rellocate(BytecodeGenerator &generator, const std::vector<size_t> &offsets)
{
	m_label = generator.label(m_label);
	const size_t offset = offsets[m_label.function_id()];
	m_label.set_position(m_label.position() + offset);
}

void Equal::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &lhs = vm.reg(m_lhs);
	const auto &rhs = vm.reg(m_rhs);

	if (auto result = equals(lhs, rhs, interpreter)) {
		ASSERT(vm.registers().size() > m_dst)
		vm.reg(m_dst) = *result;
	}
};

std::optional<Value> add(const Value &lhs, const Value &rhs, Interpreter &interpreter)
{
	auto result = std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
					   return lhs_value + rhs_value;
				   },
			[&interpreter](const std::shared_ptr<PyObject> &lhs_value,
				const std::shared_ptr<PyObject> &rhs_value) -> std::optional<Value> {
				return lhs_value->add_impl(rhs_value, interpreter);
			},
			[&interpreter](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->add_impl(py_rhs, interpreter)) { return result; }
				return {};
			} },
		lhs,
		rhs);

	return result;
}

std::optional<Value> subtract(const Value &lhs, const Value &rhs, Interpreter &interpreter)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
					   return lhs_value - rhs_value;
				   },
			[&interpreter](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->subtract_impl(py_rhs, interpreter)) { return result; }
				return {};
			} },
		lhs,
		rhs);
}

std::optional<Value> multiply(const Value &lhs, const Value &rhs, Interpreter &interpreter)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
					   return lhs_value * rhs_value;
				   },
			[&interpreter](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->multiply_impl(py_rhs, interpreter)) { return result; }
				return {};
			} },
		lhs,
		rhs);
}

std::optional<Value> exp(const Value &lhs, const Value &rhs, Interpreter &interpreter)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
					   return lhs_value.exp(rhs_value);
				   },
			[&interpreter](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->exp_impl(py_rhs, interpreter)) { return result; }
				return {};
			} },
		lhs,
		rhs);
}

std::optional<Value> lshift(const Value &lhs, const Value &rhs, Interpreter &interpreter)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
					   return lhs_value << rhs_value;
				   },
			[&interpreter](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->lshift_impl(py_rhs, interpreter)) { return result; }
				return {};
			} },
		lhs,
		rhs);
}

std::optional<Value> equals(const Value &lhs, const Value &rhs, Interpreter &interpreter)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> std::optional<Value> {
					   return NameConstant{ lhs_value == rhs_value };
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> std::optional<Value> {
				return NameConstant{ lhs_value == rhs_value };
			},
			[](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
				return NameConstant{ lhs_value == rhs_value };
			},
			[&interpreter](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->equal_impl(py_rhs, interpreter)) { return result; }
				return {};
			} },
		lhs,
		rhs);
}
