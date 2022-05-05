#include "Value.hpp"
#include "PyBool.hpp"
#include "PyBytes.hpp"
#include "PyEllipsis.hpp"
#include "PyFloat.hpp"
#include "PyInteger.hpp"
#include "PyNone.hpp"
#include "PyNumber.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "PyType.hpp"
#include "TypeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"

using namespace py;

Number Number::operator/(const Number &other) const
{
	return std::visit(
		overloaded{ [](const auto &lhs, const auto &rhs) {
					   return Number{ static_cast<double>(lhs) / static_cast<double>(rhs) };
				   },
			[](const int64_t &lhs, const int64_t &rhs) {
				if (lhs % rhs == 0) {
					return Number{ lhs / rhs };
				} else {
					return Number{ static_cast<double>(lhs) / static_cast<double>(rhs) };
				}
			} },
		value,
		other.value);
}

Number Number::operator*(const Number &other) const
{
	return std::visit(
		overloaded{ [](const auto &lhs, const auto &rhs) {
					   return Number{ static_cast<double>(lhs) * static_cast<double>(rhs) };
				   },
			[](const int64_t &lhs, const int64_t &rhs) { return Number{ lhs * rhs }; } },
		value,
		other.value);
}

bool Number::operator==(const PyObject *other) const
{
	if (auto other_pynumber = as<PyInteger>(other)) {
		return *this == other_pynumber->value();
	} else if (auto other_pynumber = as<PyFloat>(other)) {
		return *this == other_pynumber->value();
	} else {
		return false;
	}
}

bool Number::operator==(const NameConstant &other) const
{
	if (std::holds_alternative<NoneType>(other.value)) { return false; }
	if (*this == Number{ int64_t{ 0 } }) { return std::get<bool>(other.value) == false; }
	if (*this == Number{ int64_t{ 1 } }) { return std::get<bool>(other.value) == true; }
	return false;
}

bool Number::operator==(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) { return lhs_value == rhs_value; },
		value,
		rhs.value);
}
bool Number::operator<=(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) { return lhs_value <= rhs_value; },
		value,
		rhs.value);
}
bool Number::operator<(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) { return lhs_value < rhs_value; },
		value,
		rhs.value);
}

bool Number::operator>(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) { return lhs_value > rhs_value; },
		value,
		rhs.value);
}

bool Number::operator>=(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) { return lhs_value >= rhs_value; },
		value,
		rhs.value);
}


bool String::operator==(const PyObject *other) const
{
	if (auto other_pystr = as<PyString>(other)) {
		return s == other_pystr->value();
	} else {
		return false;
	}
}


bool Bytes::operator==(const PyObject *other) const
{
	if (auto other_pybytes = as<PyBytes>(other)) {
		return *this == other_pybytes->value();
	} else {
		return false;
	}
}

bool Ellipsis::operator==(const PyObject *other) const { return other == py_ellipsis(); }


bool NoneType::operator==(const PyObject *other) const { return other == py_none(); }


bool NameConstant::operator==(const PyObject *other) const
{
	if (std::holds_alternative<NoneType>(value)) { return other == py_none(); }
	const auto bool_value = std::get<bool>(value);
	if (bool_value) {
		return other == py_true();
	} else {
		return other == py_false();
	}
}


bool NameConstant::operator==(const Number &other) const
{
	if (std::holds_alternative<NoneType>(value)) { return false; }
	const short bool_value = std::get<bool>(value);
	if (bool_value) {
		return other == Number{ int64_t{ 1 } };
	} else {
		return other == Number{ int64_t{ 0 } };
	}
}

bool NameConstant::operator==(const NameConstant &other) const
{
	return std::visit(
		[](const auto &rhs, const auto &lhs) { return rhs == lhs; }, value, other.value);
}

namespace py {

PyResult<Value> add(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value + rhs_value);
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->add(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> subtract(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value - rhs_value);
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->subtract(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> multiply(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value * rhs_value);
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->multiply(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> exp(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value.exp(rhs_value));
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->exp(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> lshift(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value << rhs_value);
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->lshift(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> modulo(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value % rhs_value);
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->modulo(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> true_divide(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value / rhs_value);
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				const auto py_rhs = PyObject::from(rhs_value);
				(void)py_lhs;
				(void)py_rhs;
				// if (auto result = py_lhs->true_divide(py_rhs)) { return result; }
				TODO();
				return Ok(nullptr);
			} },
		lhs,
		rhs);
}

PyResult<Value> equals(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult<Value> {
					   return Ok(NameConstant{ lhs_value == rhs_value });
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value == rhs_value });
			},
			[](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value == rhs_value });
			},
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->richcompare(py_rhs.unwrap(), RichCompare::Py_EQ);
			} },
		lhs,
		rhs);
}

PyResult<Value> not_equals(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult<Value> {
					   return Ok(NameConstant{ lhs_value != rhs_value });
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value != rhs_value });
			},
			[](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value != rhs_value });
			},
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->richcompare(py_rhs.unwrap(), RichCompare::Py_NE);
			} },
		lhs,
		rhs);
}

PyResult<Value> less_than_equals(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult<Value> {
					   return Ok(NameConstant{ lhs_value <= rhs_value });
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value <= rhs_value });
			},
			[](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value <= rhs_value });
			},
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->richcompare(py_rhs.unwrap(), RichCompare::Py_LE);
			} },
		lhs,
		rhs);
}


PyResult<Value> less_than(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult<Value> {
					   return Ok(NameConstant{ lhs_value < rhs_value });
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value < rhs_value });
			},
			[](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value < rhs_value });
			},
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->richcompare(py_rhs.unwrap(), RichCompare::Py_LT);
			} },
		lhs,
		rhs);
}

PyResult<Value> greater_than(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult<Value> {
					   return Ok(NameConstant{ lhs_value > rhs_value });
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value > rhs_value });
			},
			[](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value > rhs_value });
			},
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->richcompare(py_rhs.unwrap(), RichCompare::Py_GT);
			} },
		lhs,
		rhs);
}

PyResult<Value> greater_than_equals(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult<Value> {
					   return Ok(NameConstant{ lhs_value >= rhs_value });
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value >= rhs_value });
			},
			[](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value >= rhs_value });
			},
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->richcompare(py_rhs.unwrap(), RichCompare::Py_GE);
			} },
		lhs,
		rhs);
}

PyResult<bool> is(const Value &lhs, const Value &rhs, Interpreter &)
{
	// TODO: Could probably be more efficient, but at least guarantees that Python singletons
	//		 always are true in this comparisson when compared to the same singleton
	auto lhs_ = PyObject::from(lhs);
	if (lhs_.is_err()) return lhs_;
	auto rhs_ = PyObject::from(rhs);
	if (rhs_.is_err()) return rhs_;

	return Ok(lhs_.unwrap() == rhs_.unwrap());
}

PyResult<bool> in(const Value &lhs, const Value &rhs, Interpreter &)
{
	TODO();
	(void)lhs;
	(void)rhs;
	return Ok(false);
}

PyResult<bool> truthy(const Value &value, Interpreter &)
{
	// Number, String, Bytes, Ellipsis, NameConstant, PyObject *
	return std::visit(overloaded{ [](const NameConstant &c) -> PyResult<bool> {
									 if (std::holds_alternative<NoneType>(c.value)) {
										 return Ok(false);
									 } else {
										 return Ok(std::get<bool>(c.value));
									 }
								 },
						  [](const Number &number) -> PyResult<bool> {
							  return Ok(number != Number{ int64_t{ 0 } });
						  },
						  [](const String &str) -> PyResult<bool> { return Ok(!str.s.empty()); },
						  [](const Bytes &bytes) -> PyResult<bool> { return Ok(!bytes.b.empty()); },
						  [](const Ellipsis &) -> PyResult<bool> { return Ok(true); },
						  [](PyObject *obj) -> PyResult<bool> { return obj->bool_(); } },
		value);
}

}// namespace py