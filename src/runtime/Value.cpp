#include "PyNumber.hpp"
#include "PyString.hpp"
#include "Value.hpp"


bool Number::operator==(const std::shared_ptr<PyObject> &other) const
{
	if (auto other_pynumber = as<PyNumber>(other)) {
		return *this == other_pynumber->value();
	} else {
		return false;
	}
}


bool Number::operator==(const NameConstant &other) const
{
	if (std::holds_alternative<NoneType>(other.value)) { return false; }
	if (*this == Number{ 0 }) { return std::get<bool>(other.value) == false; }
	if (*this == Number{ 1 }) { return std::get<bool>(other.value) == true; }
	return false;
}


bool String::operator==(const std::shared_ptr<PyObject> &other) const
{
	if (auto other_pystr = as<PyString>(other)) {
		return s == other_pystr->value();
	} else {
		return false;
	}
}


bool Bytes::operator==(const std::shared_ptr<PyObject> &other) const
{
	if (auto other_pybytes = as<PyBytes>(other)) {
		return *this == other_pybytes->value();
	} else {
		return false;
	}
}

bool Ellipsis::operator==(const std::shared_ptr<PyObject> &other) const
{
	return other == py_ellipsis();
}


bool NoneType::operator==(const std::shared_ptr<PyObject> &other) const
{
	return other == py_none();
}


bool NameConstant::operator==(const std::shared_ptr<PyObject> &other) const
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
		return other == Number{ 1 };
	} else {
		return other == Number{ 0 };
	}
}
