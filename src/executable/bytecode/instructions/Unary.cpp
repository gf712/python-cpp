#include "Unary.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/TypeError.hpp"

using namespace py;

namespace {
PyResult unary_positive(const Value &val)
{
	return std::visit(
		overloaded{ [](const Number &val) -> PyResult { return PyResult::Ok(val); },
			[](const String &) -> PyResult {
				return PyResult::Err(type_error("bad operand type for unary +: 'str'"));
			},
			[](const Bytes &) -> PyResult {
				return PyResult::Err(type_error("bad operand type for unary +: 'bytes'"));
			},
			[](const Ellipsis &) -> PyResult {
				return PyResult::Err(type_error("bad operand type for unary +: 'ellipsis'"));
			},
			[](const NameConstant &c) -> PyResult {
				if (std::holds_alternative<NoneType>(c.value)) {
					return PyResult::Err(type_error("bad operand type for unary +: 'NoneType'"));
				}
				return PyResult::Ok(c);
			},
			[](PyObject *obj) -> PyResult { return obj->pos(); } },
		val);
}

PyResult unary_negative(const Value &val)
{
	return std::visit(
		overloaded{ [](const Number &val) -> PyResult {
					   return PyResult::Ok(
						   std::visit([](const auto &v) { return Number{ -v }; }, val.value));
				   },
			[](const String &) -> PyResult {
				return PyResult::Err(type_error("bad operand type for unary -: 'str'"));
			},
			[](const Bytes &) -> PyResult {
				return PyResult::Err(type_error("bad operand type for unary -: 'bytes'"));
			},
			[](const Ellipsis &) -> PyResult {
				return PyResult::Err(type_error("bad operand type for unary -: 'ellipsis'"));
			},
			[](const NameConstant &c) -> PyResult {
				if (std::holds_alternative<NoneType>(c.value)) {
					return PyResult::Err(type_error("bad operand type for unary -: 'NoneType'"));
				}
				if (std::get<bool>(c.value)) { return PyResult::Ok(Number{ int64_t{ -1 } }); }
				return PyResult::Ok(Number{ int64_t{ 0 } });
			},
			[](PyObject *obj) -> PyResult { return obj->neg(); } },
		val);
}
}// namespace

PyResult Unary::execute(VirtualMachine &vm, Interpreter &) const
{
	const auto &val = vm.reg(m_source);
	auto result = [&]() -> PyResult {
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
			TODO();
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