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

PyResult add(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult {
									 return PyResult::Ok(lhs_value + rhs_value);
								 },
						  [](const auto &lhs_value, const auto &rhs_value) -> PyResult {
							  const auto py_lhs = PyObject::from(lhs_value);
							  if (py_lhs.is_err()) return py_lhs;
							  const auto py_rhs = PyObject::from(rhs_value);
							  if (py_rhs.is_err()) return py_rhs;
							  return py_lhs.template unwrap_as<PyObject>()->add(
								  py_rhs.template unwrap_as<PyObject>());
						  } },
		lhs,
		rhs);
}

PyResult subtract(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult {
									 return PyResult::Ok(lhs_value - rhs_value);
								 },
						  [](const auto &lhs_value, const auto &rhs_value) -> PyResult {
							  const auto py_lhs = PyObject::from(lhs_value);
							  if (py_lhs.is_err()) return py_lhs;
							  const auto py_rhs = PyObject::from(rhs_value);
							  if (py_rhs.is_err()) return py_rhs;
							  return py_lhs.template unwrap_as<PyObject>()->subtract(
								  py_rhs.template unwrap_as<PyObject>());
						  } },
		lhs,
		rhs);
}

PyResult multiply(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult {
									 return PyResult::Ok(lhs_value * rhs_value);
								 },
						  [](const auto &lhs_value, const auto &rhs_value) -> PyResult {
							  const auto py_lhs = PyObject::from(lhs_value);
							  if (py_lhs.is_err()) return py_lhs;
							  const auto py_rhs = PyObject::from(rhs_value);
							  if (py_rhs.is_err()) return py_rhs;
							  return py_lhs.template unwrap_as<PyObject>()->multiply(
								  py_rhs.template unwrap_as<PyObject>());
						  } },
		lhs,
		rhs);
}

PyResult exp(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult {
									 return PyResult::Ok(lhs_value.exp(rhs_value));
								 },
						  [](const auto &lhs_value, const auto &rhs_value) -> PyResult {
							  const auto py_lhs = PyObject::from(lhs_value);
							  if (py_lhs.is_err()) return py_lhs;
							  const auto py_rhs = PyObject::from(rhs_value);
							  if (py_rhs.is_err()) return py_rhs;
							  return py_lhs.template unwrap_as<PyObject>()->exp(
								  py_rhs.template unwrap_as<PyObject>());
						  } },
		lhs,
		rhs);
}

PyResult lshift(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult {
									 return PyResult::Ok(lhs_value << rhs_value);
								 },
						  [](const auto &lhs_value, const auto &rhs_value) -> PyResult {
							  const auto py_lhs = PyObject::from(lhs_value);
							  if (py_lhs.is_err()) return py_lhs;
							  const auto py_rhs = PyObject::from(rhs_value);
							  if (py_rhs.is_err()) return py_rhs;
							  return py_lhs.template unwrap_as<PyObject>()->lshift(
								  py_rhs.template unwrap_as<PyObject>());
						  } },
		lhs,
		rhs);
}

PyResult modulo(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult {
									 return PyResult::Ok(lhs_value % rhs_value);
								 },
						  [](const auto &lhs_value, const auto &rhs_value) -> PyResult {
							  const auto py_lhs = PyObject::from(lhs_value);
							  if (py_lhs.is_err()) return py_lhs;
							  const auto py_rhs = PyObject::from(rhs_value);
							  if (py_rhs.is_err()) return py_rhs;
							  return py_lhs.template unwrap_as<PyObject>()->modulo(
								  py_rhs.template unwrap_as<PyObject>());
						  } },
		lhs,
		rhs);
}

PyResult true_divide(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult {
									 return PyResult::Ok(lhs_value / rhs_value);
								 },
						  [](const auto &lhs_value, const auto &rhs_value) -> PyResult {
							  const auto py_lhs = PyObject::from(lhs_value);
							  const auto py_rhs = PyObject::from(rhs_value);
							  (void)py_lhs;
							  (void)py_rhs;
							  // if (auto result = py_lhs->true_divide(py_rhs)) { return result; }
							  TODO();
							  return PyResult::Ok(nullptr);
						  } },
		lhs,
		rhs);
}

PyResult equals(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult {
									 return PyResult::Ok(NameConstant{ lhs_value == rhs_value });
								 },
						  [](const auto &lhs_value, const NoneType &rhs_value) -> PyResult {
							  return PyResult::Ok(NameConstant{ lhs_value == rhs_value });
						  },
						  [](const Number &lhs_value, const Number &rhs_value) -> PyResult {
							  return PyResult::Ok(NameConstant{ lhs_value == rhs_value });
						  },
						  [](const auto &lhs_value, const auto &rhs_value) -> PyResult {
							  const auto py_lhs = PyObject::from(lhs_value);
							  if (py_lhs.is_err()) return py_lhs;
							  const auto py_rhs = PyObject::from(rhs_value);
							  if (py_rhs.is_err()) return py_rhs;
							  return py_lhs.template unwrap_as<PyObject>()->richcompare(
								  py_rhs.template unwrap_as<PyObject>(), RichCompare::Py_EQ);
						  } },
		lhs,
		rhs);
}

PyResult not_equals(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult {
									 return PyResult::Ok(NameConstant{ lhs_value != rhs_value });
								 },
						  [](const auto &lhs_value, const NoneType &rhs_value) -> PyResult {
							  return PyResult::Ok(NameConstant{ lhs_value != rhs_value });
						  },
						  [](const Number &lhs_value, const Number &rhs_value) -> PyResult {
							  return PyResult::Ok(NameConstant{ lhs_value != rhs_value });
						  },
						  [](const auto &lhs_value, const auto &rhs_value) -> PyResult {
							  const auto py_lhs = PyObject::from(lhs_value);
							  if (py_lhs.is_err()) return py_lhs;
							  const auto py_rhs = PyObject::from(rhs_value);
							  if (py_rhs.is_err()) return py_rhs;
							  return py_lhs.template unwrap_as<PyObject>()->richcompare(
								  py_rhs.template unwrap_as<PyObject>(), RichCompare::Py_NE);
						  } },
		lhs,
		rhs);
}

PyResult less_than_equals(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult {
									 return PyResult::Ok(NameConstant{ lhs_value <= rhs_value });
								 },
						  [](const auto &lhs_value, const NoneType &rhs_value) -> PyResult {
							  return PyResult::Ok(NameConstant{ lhs_value <= rhs_value });
						  },
						  [](const Number &lhs_value, const Number &rhs_value) -> PyResult {
							  return PyResult::Ok(NameConstant{ lhs_value <= rhs_value });
						  },
						  [](const auto &lhs_value, const auto &rhs_value) -> PyResult {
							  const auto py_lhs = PyObject::from(lhs_value);
							  if (py_lhs.is_err()) return py_lhs;
							  const auto py_rhs = PyObject::from(rhs_value);
							  if (py_rhs.is_err()) return py_rhs;
							  return py_lhs.template unwrap_as<PyObject>()->richcompare(
								  py_rhs.template unwrap_as<PyObject>(), RichCompare::Py_LE);
						  } },
		lhs,
		rhs);
}


PyResult less_than(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult {
									 return PyResult::Ok(NameConstant{ lhs_value < rhs_value });
								 },
						  [](const auto &lhs_value, const NoneType &rhs_value) -> PyResult {
							  return PyResult::Ok(NameConstant{ lhs_value < rhs_value });
						  },
						  [](const Number &lhs_value, const Number &rhs_value) -> PyResult {
							  return PyResult::Ok(NameConstant{ lhs_value < rhs_value });
						  },
						  [](const auto &lhs_value, const auto &rhs_value) -> PyResult {
							  const auto py_lhs = PyObject::from(lhs_value);
							  if (py_lhs.is_err()) return py_lhs;
							  const auto py_rhs = PyObject::from(rhs_value);
							  if (py_rhs.is_err()) return py_rhs;
							  return py_lhs.template unwrap_as<PyObject>()->richcompare(
								  py_rhs.template unwrap_as<PyObject>(), RichCompare::Py_LT);
						  } },
		lhs,
		rhs);
}

PyResult greater_than(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult {
									 return PyResult::Ok(NameConstant{ lhs_value > rhs_value });
								 },
						  [](const auto &lhs_value, const NoneType &rhs_value) -> PyResult {
							  return PyResult::Ok(NameConstant{ lhs_value > rhs_value });
						  },
						  [](const Number &lhs_value, const Number &rhs_value) -> PyResult {
							  return PyResult::Ok(NameConstant{ lhs_value > rhs_value });
						  },
						  [](const auto &lhs_value, const auto &rhs_value) -> PyResult {
							  const auto py_lhs = PyObject::from(lhs_value);
							  if (py_lhs.is_err()) return py_lhs;
							  const auto py_rhs = PyObject::from(rhs_value);
							  if (py_rhs.is_err()) return py_rhs;
							  return py_lhs.template unwrap_as<PyObject>()->richcompare(
								  py_rhs.template unwrap_as<PyObject>(), RichCompare::Py_GT);
						  } },
		lhs,
		rhs);
}

PyResult greater_than_equals(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult {
									 return PyResult::Ok(NameConstant{ lhs_value >= rhs_value });
								 },
						  [](const auto &lhs_value, const NoneType &rhs_value) -> PyResult {
							  return PyResult::Ok(NameConstant{ lhs_value >= rhs_value });
						  },
						  [](const Number &lhs_value, const Number &rhs_value) -> PyResult {
							  return PyResult::Ok(NameConstant{ lhs_value >= rhs_value });
						  },
						  [](const auto &lhs_value, const auto &rhs_value) -> PyResult {
							  const auto py_lhs = PyObject::from(lhs_value);
							  if (py_lhs.is_err()) return py_lhs;
							  const auto py_rhs = PyObject::from(rhs_value);
							  if (py_rhs.is_err()) return py_rhs;
							  return py_lhs.template unwrap_as<PyObject>()->richcompare(
								  py_rhs.template unwrap_as<PyObject>(), RichCompare::Py_GE);
						  } },
		lhs,
		rhs);
}

PyResult is(const Value &lhs, const Value &rhs, Interpreter &)
{
	// TODO: Could probably be more efficient, but at least guarantees that Python singletons
	//		 always are true in this comparisson when compared to the same singleton
	auto lhs_ = PyObject::from(lhs);
	if (lhs_.is_err()) return lhs_;
	auto rhs_ = PyObject::from(rhs);
	if (rhs_.is_err()) return rhs_;

	return PyResult::Ok(NameConstant{ lhs_.unwrap_as<PyObject>() == rhs_.unwrap_as<PyObject>() });
}

PyResult in(const Value &lhs, const Value &rhs, Interpreter &)
{
	TODO();
	(void)lhs;
	(void)rhs;
	return PyResult::Ok(NameConstant{ false });
}

PyResult truthy(const Value &value, Interpreter &)
{
	// Number, String, Bytes, Ellipsis, NameConstant, PyObject *
	return std::visit(
		overloaded{ [](const NameConstant &c) {
					   if (std::holds_alternative<NoneType>(c.value)) {
						   return PyResult::Ok(NameConstant{ false });
					   } else {
						   return PyResult::Ok(c);
					   }
				   },
			[](const Number &number) {
				return PyResult::Ok(NameConstant{ number != Number{ int64_t{ 0 } } });
			},
			[](const String &str) { return PyResult::Ok(NameConstant{ !str.s.empty() }); },
			[](const Bytes &bytes) { return PyResult::Ok(NameConstant{ !bytes.b.empty() }); },
			[](const Ellipsis &) { return PyResult::Ok(NameConstant{ true }); },
			[](PyObject *obj) { return obj->bool_(); } },
		value);
}

}// namespace py