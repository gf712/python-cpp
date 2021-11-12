#include "Instructions.hpp"

#include "executable/bytecode/Bytecode.hpp"

#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyNumber.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/StopIterationException.hpp"


void MakeFunction::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(interpreter.functions(m_function_id)->backend() == FunctionExecutionBackend::BYTECODE)
	auto function = std::static_pointer_cast<Bytecode>(interpreter.functions(m_function_id));
	auto *code = vm.heap().allocate<PyCode>(function, m_function_id, m_args, interpreter.module());
	interpreter.allocate_object<PyFunction>(
		m_function_name, m_function_name, code, interpreter.execution_frame()->globals());
}


void JumpIfFalse::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &result = vm.reg(m_test_register);

	const bool test_result =
		std::visit(overloaded{ [](PyObject *const &obj) -> bool {
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
	if (!test_result) {
		const auto ip = vm.instruction_pointer() + m_label.position();
		vm.set_instruction_pointer(ip);
	}
};


void JumpIfFalse::relocate(BytecodeGenerator &generator, size_t instruction_idx)
{
	m_label = generator.label(m_label);
	m_label.set_position(m_label.position() - instruction_idx - 1);
}

void Jump::execute(VirtualMachine &vm, Interpreter &) const
{
	const auto ip = vm.instruction_pointer() + m_label.position();
	vm.set_instruction_pointer(ip);
};


void Jump::relocate(BytecodeGenerator &generator, size_t instruction_idx)
{
	m_label = generator.label(m_label);
	m_label.set_position(m_label.position() - instruction_idx - 1);
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

void LessThanEquals::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &lhs = vm.reg(m_lhs);
	const auto &rhs = vm.reg(m_rhs);

	if (auto result = less_than_equals(lhs, rhs, interpreter)) {
		ASSERT(vm.registers().size() > m_dst)
		vm.reg(m_dst) = *result;
	}
};

void LessThan::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &lhs = vm.reg(m_lhs);
	const auto &rhs = vm.reg(m_rhs);

	if (auto result = less_than(lhs, rhs, interpreter)) {
		ASSERT(vm.registers().size() > m_dst)
		vm.reg(m_dst) = *result;
	}
};

void BuildList::execute(VirtualMachine &vm, Interpreter &) const
{
	std::vector<Value> elements;
	for (const auto &src : m_srcs) { elements.push_back(vm.reg(src)); }

	auto &heap = vm.heap();
	vm.reg(m_dst) = heap.allocate<PyList>(elements);
};


void BuildTuple::execute(VirtualMachine &vm, Interpreter &) const
{
	std::vector<Value> elements;
	for (const auto &src : m_srcs) { elements.push_back(vm.reg(src)); }

	vm.reg(m_dst) = PyTuple::create(elements);
};

void BuildDict::execute(VirtualMachine &vm, Interpreter &) const
{
	PyDict::MapType map;
	for (size_t i = 0; i < m_keys.size(); ++i) {
		map.emplace(vm.reg(m_keys[i]), vm.reg(m_values[i]));
	}

	auto &heap = vm.heap();
	vm.reg(m_dst) = heap.allocate<PyDict>(map);
};

void GetIter::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto iterable_value = vm.reg(m_src);
	if (auto *iterable_object = std::get_if<PyObject *>(&iterable_value)) {
		vm.reg(m_dst) = (*iterable_object)->iter_impl(interpreter);
	} else {
		vm.reg(m_dst) = std::visit(
			[&interpreter](
				const auto &value) { return PyObject::from(value)->iter_impl(interpreter); },
			iterable_value);
	}
}

void ForIter::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto iterator = vm.reg(m_src);
	interpreter.execution_frame()->set_exception_to_catch(stop_iteration(""));
	if (auto *iterable_object = std::get_if<PyObject *>(&iterator)) {
		const auto &next_value = (*iterable_object)->next_impl(interpreter);
		if (auto last_exception = interpreter.execution_frame()->exception()) {
			if (!interpreter.execution_frame()->catch_exception(last_exception)) {
				// exit loop in error state and handle unwinding to interpreter
			} else {
				interpreter.execution_frame()->set_exception(nullptr);
				interpreter.set_status(Interpreter::Status::OK);
				vm.set_instruction_pointer(vm.instruction_pointer() + m_exit_label.position());
			}
			return;
		}
		interpreter.store_object(m_next_value_name, next_value);
	} else {
		// this is probably always going to be something that went wrong
		TODO();
	}
}

void ForIter::relocate(BytecodeGenerator &generator, size_t instruction_idx)
{
	m_exit_label = generator.label(m_exit_label);
	m_exit_label.set_position(m_exit_label.position() - instruction_idx);
}

std::optional<Value> add(const Value &lhs, const Value &rhs, Interpreter &interpreter)
{
	auto result = std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
					   return lhs_value + rhs_value;
				   },
			[&interpreter](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->add_impl(py_rhs, interpreter)) { return result; }
				interpreter.raise_exception(
					"TypeError: unsupported operand type(s) for +: \'{}\' and \'{}\'",
					object_name(py_lhs->type()),
					object_name(py_rhs->type()));
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
				interpreter.raise_exception(
					"TypeError: unsupported operand type(s) for -: \'{}\' and \'{}\'",
					object_name(py_lhs->type()),
					object_name(py_rhs->type()));
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
				interpreter.raise_exception(
					"TypeError: unsupported operand type(s) for *: \'{}\' and \'{}\'",
					object_name(py_lhs->type()),
					object_name(py_rhs->type()));
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
				interpreter.raise_exception(
					"TypeError: unsupported operand type(s) for **: \'{}\' and \'{}\'",
					object_name(py_lhs->type()),
					object_name(py_rhs->type()));
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
				interpreter.raise_exception(
					"TypeError: unsupported operand type(s) for <<: \'{}\' and \'{}\'",
					object_name(py_lhs->type()),
					object_name(py_rhs->type()));
				return {};
			} },
		lhs,
		rhs);
}

std::optional<Value> modulo(const Value &lhs, const Value &rhs, Interpreter &interpreter)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
					   return lhs_value % rhs_value;
				   },
			[&interpreter](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->modulo_impl(py_rhs, interpreter)) { return result; }
				interpreter.raise_exception(
					"TypeError: unsupported operand type(s) for %: \'{}\' and \'{}\'",
					object_name(py_lhs->type()),
					object_name(py_rhs->type()));
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
				if (auto result = py_lhs->equal_impl(py_rhs, interpreter)) {
					return result;
				} else {
					interpreter.raise_exception(
						"TypeError: unsupported operand type(s) for ==: \'{}\' and \'{}\'",
						object_name(py_lhs->type()),
						object_name(py_rhs->type()));
					return {};
				}
			} },
		lhs,
		rhs);
}

std::optional<Value> less_than_equals(const Value &lhs, const Value &rhs, Interpreter &interpreter)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> std::optional<Value> {
					   return NameConstant{ lhs_value <= rhs_value };
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> std::optional<Value> {
				return NameConstant{ lhs_value <= rhs_value };
			},
			[](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
				return NameConstant{ lhs_value <= rhs_value };
			},
			[&interpreter](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->less_than_equal_impl(py_rhs, interpreter)) {
					return result;
				} else {
					interpreter.raise_exception(
						"TypeError: unsupported operand type(s) for <=: \'{}\' and \'{}\'",
						object_name(py_lhs->type()),
						object_name(py_rhs->type()));
					return {};
				}
			} },
		lhs,
		rhs);
}


std::optional<Value> less_than(const Value &lhs, const Value &rhs, Interpreter &interpreter)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> std::optional<Value> {
					   return NameConstant{ lhs_value < rhs_value };
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> std::optional<Value> {
				return NameConstant{ lhs_value < rhs_value };
			},
			[](const Number &lhs_value, const Number &rhs_value) -> std::optional<Value> {
				return NameConstant{ lhs_value < rhs_value };
			},
			[&interpreter](const auto &lhs_value, const auto &rhs_value) -> std::optional<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				if (auto result = py_lhs->less_than_impl(py_rhs, interpreter)) {
					return result;
				} else {
					interpreter.raise_exception(
						"TypeError: unsupported operand type(s) for <: \'{}\' and \'{}\'",
						object_name(py_lhs->type()),
						object_name(py_rhs->type()));
					return {};
				}
			} },
		lhs,
		rhs);
}