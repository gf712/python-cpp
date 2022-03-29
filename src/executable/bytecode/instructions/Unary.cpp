#include "Unary.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/TypeError.hpp"

using namespace py;

namespace {
Value unary_positive(const Value &val, Interpreter &interpreter)
{
	return std::visit(
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
}

Value unary_negative(const Value &val, Interpreter &interpreter)
{
	return std::visit(
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
}
}// namespace

void Unary::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	const auto &val = vm.reg(m_source);
	switch (m_operation) {
	case Operation::POSITIVE: {
		vm.reg(m_destination) = unary_positive(val, interpreter);
	} break;
	case Operation::NEGATIVE: {
		vm.reg(m_destination) = unary_negative(val, interpreter);
	} break;
	case Operation::INVERT: {
		TODO();
	} break;
	case Operation::NOT: {
		TODO();
	} break;
	}
}

std::vector<uint8_t> Unary::serialize() const
{
	return {
		UNARY,
		m_destination,
		m_source,
		static_cast<uint8_t>(m_operation),
	};
}