#include "Instructions.hpp"

#include "executable/bytecode/Bytecode.hpp"

#include "runtime/PyBool.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyNumber.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/StopIterationException.hpp"


void Move::execute(VirtualMachine &vm, Interpreter &) const
{
	vm.reg(m_destination) = vm.reg(m_source);
}

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

	const bool test_result = std::visit(
		overloaded{ [](PyObject *const &obj) -> bool { return obj->bool_() == py_true(); },
			[](const auto &) -> bool {
				TODO();
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
		const auto ip = vm.instruction_pointer() + m_label->position();
		vm.set_instruction_pointer(ip);
	}
};


void JumpIfFalse::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_label->set_position(m_label->position() - instruction_idx - 1);
	m_label->immutable();
}

void Jump::execute(VirtualMachine &vm, Interpreter &) const
{
	const auto ip = vm.instruction_pointer() + m_label->position();
	vm.set_instruction_pointer(ip);
};


void Jump::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_label->set_position(m_label->position() - instruction_idx - 1);
	m_label->immutable();
}

void Equal::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &lhs = vm.reg(m_lhs);
	const auto &rhs = vm.reg(m_rhs);

	if (auto result = equals(lhs, rhs, interpreter)) {
		ASSERT(vm.registers().has_value())
		ASSERT(vm.registers()->get().size() > m_dst)
		vm.reg(m_dst) = *result;
	}
};

void LessThanEquals::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &lhs = vm.reg(m_lhs);
	const auto &rhs = vm.reg(m_rhs);

	if (auto result = less_than_equals(lhs, rhs, interpreter)) {
		ASSERT(vm.registers().has_value())
		ASSERT(vm.registers()->get().size() > m_dst)
		vm.reg(m_dst) = *result;
	}
};

void LessThan::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &lhs = vm.reg(m_lhs);
	const auto &rhs = vm.reg(m_rhs);

	if (auto result = less_than(lhs, rhs, interpreter)) {
		ASSERT(vm.registers().has_value())
		ASSERT(vm.registers()->get().size() > m_dst)
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

void GetIter::execute(VirtualMachine &vm, Interpreter &) const
{
	auto iterable_value = vm.reg(m_src);
	if (auto *iterable_object = std::get_if<PyObject *>(&iterable_value)) {
		vm.reg(m_dst) = (*iterable_object)->iter();
	} else {
		vm.reg(m_dst) = std::visit(
			[](const auto &value) { return PyObject::from(value)->iter(); }, iterable_value);
	}
}

void ForIter::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto iterator = vm.reg(m_src);
	interpreter.execution_frame()->set_exception_to_catch(stop_iteration(""));
	if (auto *iterable_object = std::get_if<PyObject *>(&iterator)) {
		const auto &next_value = (*iterable_object)->next();
		if (auto last_exception = interpreter.execution_frame()->exception()) {
			if (!interpreter.execution_frame()->catch_exception(last_exception)) {
				// exit loop in error state and handle unwinding to interpreter
			} else {
				interpreter.execution_frame()->set_exception(nullptr);
				interpreter.set_status(Interpreter::Status::OK);
				// FIXME: subtract one since the vm will advance the ip by one.
				//        is this always true?
				vm.set_instruction_pointer(vm.instruction_pointer() + m_exit_label->position() - 1);
			}
			return;
		}
		interpreter.store_object(m_next_value_name, next_value);
	} else {
		// this is probably always going to be something that went wrong
		TODO();
	}
}

void ForIter::relocate(codegen::BytecodeGenerator &, size_t instruction_idx)
{
	m_exit_label->set_position(m_exit_label->position() - instruction_idx);
}
