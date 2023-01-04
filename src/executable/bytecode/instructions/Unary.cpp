#include "Unary.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/TypeError.hpp"
#include "vm/VM.hpp"

using namespace py;

namespace {
PyResult<Value> unary_positive(const Value &val)
{
	return std::visit(
		overloaded{ [](const Number &val) -> PyResult<Value> { return Ok(Value{ val }); },
			[](const String &) -> PyResult<Value> {
				return Err(type_error("bad operand type for unary +: 'str'"));
			},
			[](const Bytes &) -> PyResult<Value> {
				return Err(type_error("bad operand type for unary +: 'bytes'"));
			},
			[](const Ellipsis &) -> PyResult<Value> {
				return Err(type_error("bad operand type for unary +: 'ellipsis'"));
			},
			[](const NameConstant &c) -> PyResult<Value> {
				if (std::holds_alternative<NoneType>(c.value)) {
					return Err(type_error("bad operand type for unary +: 'NoneType'"));
				}
				return Ok(Value{ c });
			},
			[](PyObject *obj) -> PyResult<Value> {
				if (auto r = obj->pos(); r.is_ok()) {
					return Ok(Value{ r.unwrap() });
				} else {
					return Err(r.unwrap_err());
				}
			} },
		val);
}

PyResult<Value> unary_negative(const Value &val)
{
	return std::visit(
		overloaded{ [](const Number &val) -> PyResult<Value> {
					   return Ok(std::visit(
						   [](const auto &v) { return Value{ Number{ -v } }; }, val.value));
				   },
			[](const String &) -> PyResult<Value> {
				return Err(type_error("bad operand type for unary -: 'str'"));
			},
			[](const Bytes &) -> PyResult<Value> {
				return Err(type_error("bad operand type for unary -: 'bytes'"));
			},
			[](const Ellipsis &) -> PyResult<Value> {
				return Err(type_error("bad operand type for unary -: 'ellipsis'"));
			},
			[](const NameConstant &c) -> PyResult<Value> {
				if (std::holds_alternative<NoneType>(c.value)) {
					return Err(type_error("bad operand type for unary -: 'NoneType'"));
				}
				if (std::get<bool>(c.value)) { return Ok(Value{ Number{ int64_t{ -1 } } }); }
				return Ok(Value{ Number{ int64_t{ 0 } } });
			},
			[](PyObject *obj) -> PyResult<Value> {
				if (auto r = obj->neg(); r.is_ok()) {
					return Ok(Value{ r.unwrap() });
				} else {
					return Err(r.unwrap_err());
				}
			} },
		val);
}

PyResult<Value> unary_not(const Value &val)
{
	return std::visit(
		overloaded{ [](const Number &val) -> PyResult<Value> {
					   return Ok(std::visit(
						   [](const auto &v) { return NameConstant{ !static_cast<bool>(v) }; },
						   val.value));
				   },
			[](const String &s) -> PyResult<Value> { return Ok(NameConstant{ !s.s.empty() }); },
			[](const Bytes &) -> PyResult<Value> { TODO(); },
			[](const Ellipsis &) -> PyResult<Value> { return Ok(NameConstant{ false }); },
			[](const NameConstant &c) -> PyResult<Value> {
				if (std::holds_alternative<NoneType>(c.value)) { return Ok(NameConstant{ true }); }
				if (std::get<bool>(c.value)) { return Ok(NameConstant{ false }); }
				return Ok(NameConstant{ true });
			},
			[](PyObject *obj) -> PyResult<Value> {
				if (auto r = obj->bool_(); r.is_ok()) {
					return Ok(NameConstant{ !r.unwrap() });
				} else {
					return Err(r.unwrap_err());
				}
			} },
		val);
}

}// namespace

PyResult<Value> Unary::execute(VirtualMachine &vm, Interpreter &) const
{
	const auto &val = vm.reg(m_source);
	auto result = [&]() -> PyResult<Value> {
		switch (m_operation) {
		case Operation::POSITIVE: {
			return unary_positive(val);
		} break;
		case Operation::NEGATIVE: {
			return unary_negative(val);
		} break;
		case Operation::INVERT: {
			TODO();
		} break;
		case Operation::NOT: {
			return unary_not(val);
		} break;
		}
	}();

	if (result.is_err()) return result;
	vm.reg(m_destination) = result.unwrap();
	return result;
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