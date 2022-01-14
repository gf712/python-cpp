#include "Unary.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/TypeError.hpp"

using namespace py;

void UnaryPositive::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &val = vm.reg(m_source);
	const auto result = std::visit(
		overloaded{ [](const Number &val) -> Value { return val; },
			[&interpreter](const String &) -> Value {
				interpreter.raise_exception(type_error("bad operand type for unary +: 'str'"));
				return nullptr;
			},
			[&interpreter](const Bytes &) -> Value {
				interpreter.raise_exception(type_error("bad operand type for unary +: 'bytes'"));
				return nullptr;
			},
			[&interpreter](const Ellipsis &) -> Value {
				interpreter.raise_exception(type_error("bad operand type for unary +: 'ellipsis'"));
				return nullptr;
			},
			[&interpreter](const NameConstant &c) -> Value {
				if (std::holds_alternative<NoneType>(c.value)) {
					interpreter.raise_exception(
						type_error("bad operand type for unary +: 'NoneType'"));
					return nullptr;
				}
				return c;
			},
			[](PyObject *obj) -> Value { return obj->pos(); } },
		val);
	vm.reg(m_destination) = result;
}

void UnaryNegative::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &val = vm.reg(m_source);
	const auto result = std::visit(
		overloaded{ [](const Number &val) -> Value {
					   return std::visit([](const auto &v) { return Number{ -v }; }, val.value);
				   },
			[&interpreter](const String &) -> Value {
				interpreter.raise_exception(type_error("bad operand type for unary -: 'str'"));
				return nullptr;
			},
			[&interpreter](const Bytes &) -> Value {
				interpreter.raise_exception(type_error("bad operand type for unary -: 'bytes'"));
				return nullptr;
			},
			[&interpreter](const Ellipsis &) -> Value {
				interpreter.raise_exception(type_error("bad operand type for unary -: 'ellipsis'"));
				return nullptr;
			},
			[&interpreter](const NameConstant &c) -> Value {
				if (std::holds_alternative<NoneType>(c.value)) {
					interpreter.raise_exception(
						type_error("bad operand type for unary -: 'NoneType'"));
					return nullptr;
				}
				if (std::get<bool>(c.value)) { return Number{ int64_t{ -1 } }; }
				return Number{ int64_t{ 0 } };
			},
			[](PyObject *obj) -> Value { return obj->neg(); } },
		val);
	vm.reg(m_destination) = result;
}