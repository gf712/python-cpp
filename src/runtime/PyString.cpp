#include "PyString.hpp"

#include "interpreter/Interpreter.hpp"

std::shared_ptr<PyObject> PyString::add_impl(const std::shared_ptr<PyObject> &obj,
	Interpreter &interpreter) const
{
	if (auto rhs = as<PyString>(obj)) {
		return PyString::create(m_value + rhs->value());
	} else {

		interpreter.raise_exception(
			"TypeError: unsupported operand type(s) for +: \'{}\' and \'{}\'",
			object_name(type()),
			object_name(obj->type()));
		return nullptr;
	}
}


PyString::PyString(std::string s) : PyObject(PyObjectType::PY_STRING), m_value(std::move(s))
{
	m_slots.hash = [this]() { return this->hash_impl(*VirtualMachine::the().interpreter()); };
	m_slots.richcompare = [this](const std::shared_ptr<PyObject> &other, RichCompare op) {
		return this->richcompare_impl(other, op, *VirtualMachine::the().interpreter());
	};
}

size_t PyString::hash_impl(Interpreter &) const { return std::hash<std::string>{}(m_value); }

std::shared_ptr<PyObject> PyString::repr_impl(Interpreter &) const
{
	return PyString::from(String{ m_value });
}

std::shared_ptr<PyObject> PyString::equal_impl(const std::shared_ptr<PyObject> &obj,
	Interpreter &interpreter) const
{
	if (auto obj_string = as<PyString>(obj)) {
		return m_value == obj_string->value() ? py_true() : py_false();
	} else {
		return PyObject::equal_impl(obj, interpreter);
	}
}


std::shared_ptr<PyObject> PyString::richcompare_impl(const std::shared_ptr<PyObject> &other,
	RichCompare op,
	Interpreter &interpreter) const
{
	spdlog::debug("PyString::richcompare_impl: Compare {} to {} using {} op",
		to_string(),
		other->to_string(),
		static_cast<int>(op));
	if (auto obj_string = as<PyString>(other)) {
		switch (op) {
		case RichCompare::Py_LT: {
			return m_value < obj_string->value() ? py_true() : py_false();
		}
		case RichCompare::Py_LE: {
			return m_value <= obj_string->value() ? py_true() : py_false();
		}
		case RichCompare::Py_EQ: {
			if (this == obj_string.get()) return py_true();
			return m_value == obj_string->value() ? py_true() : py_false();
		}
		case RichCompare::Py_NE: {
			return m_value != obj_string->value() ? py_true() : py_false();
		}
		case RichCompare::Py_GT: {
			return m_value > obj_string->value() ? py_true() : py_false();
		}
		case RichCompare::Py_GE: {
			return m_value >= obj_string->value() ? py_true() : py_false();
		}
		}
		return m_value == obj_string->value() ? py_true() : py_false();
	} else {
		return PyObject::richcompare_impl(other, op, interpreter);
	}
}