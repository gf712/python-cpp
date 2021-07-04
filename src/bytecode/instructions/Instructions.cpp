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
				return std::static_pointer_cast<PyObject>(PyNameConstant::create(s));
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
