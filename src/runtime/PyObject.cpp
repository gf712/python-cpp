#include "PyObject.hpp"

#include "AttributeError.hpp"
#include "PyBool.hpp"
#include "PyBoundMethod.hpp"
#include "PyBuiltInMethod.hpp"
#include "PyBytes.hpp"
#include "PyDict.hpp"
#include "PyEllipsis.hpp"
#include "PyFunction.hpp"
#include "PyNone.hpp"
#include "PyNumber.hpp"
#include "PySlotWrapper.hpp"
#include "PyStaticMethod.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "PyType.hpp"
#include "StopIterationException.hpp"
#include "TypeError.hpp"

#include "executable/bytecode/instructions/FunctionCall.hpp"
#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"


size_t ValueHash::operator()(const Value &value) const
{
	const auto result =
		std::visit(overloaded{ [](const Number &number) -> size_t {
								  if (std::holds_alternative<double>(number.value)) {
									  return std::hash<double>{}(std::get<double>(number.value));
								  } else {
									  return std::hash<int64_t>{}(std::get<int64_t>(number.value));
								  }
							  },
					   [](const String &s) -> size_t { return std::hash<std::string>{}(s.s); },
					   [](const Bytes &b) -> size_t { return ::bit_cast<size_t>(b.b.data()); },
					   [](const Ellipsis &) -> size_t { return ::bit_cast<size_t>(py_ellipsis()); },
					   [](const NameConstant &c) -> size_t {
						   if (std::holds_alternative<bool>(c.value)) {
							   return std::get<bool>(c.value) ? 0 : 1;
						   } else {
							   return bit_cast<size_t>(py_none());
						   }
					   },
					   [](PyObject *obj) -> size_t { return obj->hash(); } },
			value);
	return result;
}


bool ValueEqual::operator()(const Value &lhs_value, const Value &rhs_value) const
{
	const auto result =
		std::visit(overloaded{ [](PyObject *const lhs, PyObject *const rhs) {
								  return lhs->richcompare(rhs, RichCompare::Py_EQ) == py_true();
							  },
					   [](PyObject *const lhs, const auto &rhs) { return lhs == rhs; },
					   [](const auto &lhs, PyObject *const rhs) { return lhs == rhs; },
					   [](const auto &lhs, const auto &rhs) { return lhs == rhs; } },
			lhs_value,
			rhs_value);
	return result;
}


template<> PyObject *PyObject::from(PyObject *const &value)
{
	ASSERT(value)
	return value;
}

template<> PyObject *PyObject::from(const Number &value) { return PyNumber::create(value); }

template<> PyObject *PyObject::from(const String &value) { return PyString::create(value.s); }

template<> PyObject *PyObject::from(const Bytes &value) { return PyBytes::create(value); }

template<> PyObject *PyObject::from(const Ellipsis &) { return py_ellipsis(); }

template<> PyObject *PyObject::from(const NameConstant &value)
{
	if (std::holds_alternative<NoneType>(value.value)) {
		return py_none();
	} else {
		const bool bool_value = std::get<bool>(value.value);
		return bool_value ? py_true() : py_false();
	}
}

template<> PyObject *PyObject::from(const Value &value)
{
	return std::visit([](const auto &v) { return PyObject::from(v); }, value);
}

PyObject::PyObject(const TypePrototype &type) : Cell(), m_type_prototype(type)
{
	for (auto m : m_type_prototype.__methods__) {
		auto *builtin_method = VirtualMachine::the().heap().allocate<PyBuiltInMethod>(
			m.name,
			[this, method = m.method](
				PyTuple *args, PyDict *kwargs) { return method(this, args, kwargs); },
			this);
		put(m.name, builtin_method);
	}

	if (m_type_prototype.__dict__) {
		for (auto [attr_name, attr] : m_type_prototype.__dict__->map()) {
			if (std::get<PyObject *>(attr)) {
				auto *attr_obj = std::get<PyObject *>(attr);
				if (attr_obj->m_type_prototype.__call__.has_value()) {
					// attribute is a callable so it becomes a method
					if (as<PyFunction>(attr_obj)) {
						attr = PyBoundMethod::create(this, as<PyFunction>(attr_obj));
					} else if (as<PyStaticMethod>(attr_obj)) {
						const auto &name = as<PyStaticMethod>(attr_obj)->static_method_name();
						attr = VirtualMachine::the().heap().allocate<PyNativeFunction>(
							name->value(),
							[attr_obj](PyTuple *args, PyDict *kwargs) {
								return as<PyStaticMethod>(attr_obj)->call_static_method(
									args, kwargs);
							},
							attr_obj);
					} else if (as<PySlotWrapper>(attr_obj)) {
						const auto &name = as<PySlotWrapper>(attr_obj)->slot_name();
						attr = VirtualMachine::the().heap().allocate<PyNativeFunction>(
							name->value(),
							[attr_obj](PyTuple *args, PyDict *kwargs) {
								return as<PySlotWrapper>(attr_obj)->__call__(args, kwargs);
							},
							attr_obj);
					} else {
						TODO()
					}
				}
			}
			if (std::holds_alternative<String>(attr_name)) {
				put(std::get<String>(attr_name).s, PyObject::from(attr));
			} else if (std::holds_alternative<PyObject *>(attr_name)) {
				auto *obj = std::get<PyObject *>(attr_name);
				ASSERT(as<PyString>(obj))
				put(as<PyString>(obj)->value(), PyObject::from(attr));
			} else {
				TODO()
			}
		}
	}
}

void PyObject::visit_graph(Visitor &visitor)
{
	for (const auto &[name, obj] : m_attributes) {
		spdlog::trace("PyObject::visit_graph: {}", name);
		visitor.visit(*obj);
	}
	visitor.visit(*this);
}


void PyObject::put(std::string name, PyObject *value)
{
	m_attributes.insert_or_assign(name, value);
}


PyObject *PyObject::get(std::string name, Interpreter &interpreter) const
{
	if (auto it = m_attributes.find(name); it != m_attributes.end()) { return it->second; }
	interpreter.raise_exception(
		attribute_error(fmt::format("'{}' object has no attribute '{}'", type()->name(), name)));
	return nullptr;
}


PyObject *PyObject::__repr__() const
{
	return PyString::from(String{ fmt::format("<object at {}>", static_cast<const void *>(this)) });
}


PyObject *PyObject::richcompare(const PyObject *other, RichCompare op) const
{
	constexpr std::array opstr{ "<", "<=", "==", "!=", ">", ">=" };

	switch (op) {
	case RichCompare::Py_EQ: {
		if (auto result = eq(other); std::holds_alternative<PyObject *>(result)) {
			return std::get<PyObject *>(result);
		} else if (auto result = other->eq(this); std::holds_alternative<PyObject *>(result)) {
			return std::get<PyObject *>(result);
		}
	} break;
	case RichCompare::Py_GE: {
		if (auto result = ge(other); std::holds_alternative<PyObject *>(result)) {
			return std::get<PyObject *>(result);
		} else if (auto result = other->le(this); std::holds_alternative<PyObject *>(result)) {
			return std::get<PyObject *>(result);
		}
	} break;
	case RichCompare::Py_GT: {
		if (auto result = gt(other); std::holds_alternative<PyObject *>(result)) {
			return std::get<PyObject *>(result);
		} else if (auto result = other->lt(this); std::holds_alternative<PyObject *>(result)) {
			return std::get<PyObject *>(result);
		}
	} break;
	case RichCompare::Py_LE: {
		if (auto result = le(other); std::holds_alternative<PyObject *>(result)) {
			return std::get<PyObject *>(result);
		} else if (auto result = other->ge(this); std::holds_alternative<PyObject *>(result)) {
			return std::get<PyObject *>(result);
		}
	} break;
	case RichCompare::Py_LT: {
		if (auto result = lt(other); std::holds_alternative<PyObject *>(result)) {
			return std::get<PyObject *>(result);
		} else if (auto result = other->gt(this); std::holds_alternative<PyObject *>(result)) {
			return std::get<PyObject *>(result);
		}
	} break;
	case RichCompare::Py_NE: {
		if (auto result = ne(other); std::holds_alternative<PyObject *>(result)) {
			return std::get<PyObject *>(result);
		} else if (auto result = other->ne(this); std::holds_alternative<PyObject *>(result)) {
			return std::get<PyObject *>(result);
		}
	} break;
	}

	switch (op) {
	case RichCompare::Py_EQ: {
		return this == other ? py_true() : py_false();
	} break;
	case RichCompare::Py_NE: {
		return this != other ? py_true() : py_false();
	} break;
	default:
		// op not supported
		VirtualMachine::the().interpreter().raise_exception(
			type_error("'{}' not supported between instances of '{}' and '{}'",
				opstr[static_cast<size_t>(op)],
				m_type_prototype.__name__,
				other->m_type_prototype.__name__));
	}

	return nullptr;
}

PyObject::PyResult PyObject::eq(const PyObject *other) const
{
	if (m_type_prototype.__eq__.has_value()) {
		return m_type_prototype.__eq__->operator()(this, other);
	}
	return NotImplemented_{};
}

PyObject::PyResult PyObject::ge(const PyObject *other) const
{
	if (m_type_prototype.__eq__.has_value()) {
		return m_type_prototype.__eq__->operator()(this, other);
	}
	return NotImplemented_{};
}

PyObject::PyResult PyObject::gt(const PyObject *other) const
{
	if (m_type_prototype.__gt__.has_value()) {
		return m_type_prototype.__gt__->operator()(this, other);
	}
	return NotImplemented_{};
}

PyObject::PyResult PyObject::le(const PyObject *other) const
{
	if (m_type_prototype.__le__.has_value()) {
		return m_type_prototype.__le__->operator()(this, other);
	}
	return NotImplemented_{};
}

PyObject::PyResult PyObject::lt(const PyObject *other) const
{
	if (m_type_prototype.__lt__.has_value()) {
		return m_type_prototype.__lt__->operator()(this, other);
	}
	return NotImplemented_{};
}

PyObject::PyResult PyObject::ne(const PyObject *other) const
{
	if (m_type_prototype.__ne__.has_value()) {
		return m_type_prototype.__ne__->operator()(this, other);
	}
	return NotImplemented_{};
}

PyObject *PyObject::repr() const
{
	if (auto it = m_attributes.find("__repr__"); it != m_attributes.end()) {
		if (auto result = it->second->call(PyTuple::create(), PyDict::create())) {
			return result;
		} else {
			TODO()
		}
	} else if (m_type_prototype.__repr__.has_value()) {
		return m_type_prototype.__repr__->operator()(this);
	}
	TODO()
}

size_t PyObject::hash() const
{
	if (auto it = m_attributes.find("__hash__"); it != m_attributes.end()) {
		TODO()
	} else if (m_type_prototype.__hash__.has_value()) {
		return m_type_prototype.__hash__->operator()(this);
	} else {
		return bit_cast<size_t>(this);
	}
}

PyObject *PyObject::call(PyTuple *args, PyDict *kwargs)
{
	if (m_type_prototype.__call__.has_value()) {
		return m_type_prototype.__call__->operator()(this, args, kwargs);
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("'{}' object is not callable", m_type_prototype.__name__));
	return nullptr;
}


PyObject *PyObject::add(const PyObject *other) const
{
	if (m_type_prototype.__add__.has_value()) {
		return m_type_prototype.__add__->operator()(this, other);
	} else if (other->m_type_prototype.__add__.has_value()) {
		return other->m_type_prototype.__add__->operator()(other, this);
	}
	VirtualMachine::the().interpreter().raise_exception(
		"TypeError: unsupported operand type(s) for +: \'{}\' and \'{}\'",
		m_type_prototype.__name__,
		other->m_type_prototype.__name__);
	return nullptr;
}

PyObject *PyObject::subtract(const PyObject *other) const
{
	if (m_type_prototype.__sub__.has_value()) {
		return m_type_prototype.__sub__->operator()(this, other);
	}
	VirtualMachine::the().interpreter().raise_exception(
		"TypeError: unsupported operand type(s) for -: \'{}\' and \'{}\'",
		m_type_prototype.__name__,
		other->m_type_prototype.__name__);
	return nullptr;
}

PyObject *PyObject::multiply(const PyObject *other) const
{
	if (m_type_prototype.__mul__.has_value()) {
		return m_type_prototype.__mul__->operator()(this, other);
	} else if (other->m_type_prototype.__mul__.has_value()) {
		return other->m_type_prototype.__mul__->operator()(other, this);
	}
	VirtualMachine::the().interpreter().raise_exception(
		"TypeError: unsupported operand type(s) for *: \'{}\' and \'{}\'",
		m_type_prototype.__name__,
		other->m_type_prototype.__name__);
	return nullptr;
}

PyObject *PyObject::exp(const PyObject *other) const
{
	if (m_type_prototype.__exp__.has_value()) {
		return m_type_prototype.__exp__->operator()(this, other);
	}
	VirtualMachine::the().interpreter().raise_exception(
		"TypeError: unsupported operand type(s) for **: \'{}\' and \'{}\'",
		m_type_prototype.__name__,
		other->m_type_prototype.__name__);
	return nullptr;
}

PyObject *PyObject::lshift(const PyObject *other) const
{
	if (m_type_prototype.__lshift__.has_value()) {
		return m_type_prototype.__lshift__->operator()(this, other);
	}
	VirtualMachine::the().interpreter().raise_exception(
		"TypeError: unsupported operand type(s) for <<: \'{}\' and \'{}\'",
		m_type_prototype.__name__,
		other->m_type_prototype.__name__);
	return nullptr;
}

PyObject *PyObject::modulo(const PyObject *other) const
{
	if (m_type_prototype.__mod__.has_value()) {
		return m_type_prototype.__mod__->operator()(this, other);
	}
	VirtualMachine::the().interpreter().raise_exception(
		"TypeError: unsupported operand type(s) for %: \'{}\' and \'{}\'",
		m_type_prototype.__name__,
		other->m_type_prototype.__name__);
	return nullptr;
}

PyObject *PyObject::abs() const
{
	if (m_type_prototype.__abs__.has_value()) { return m_type_prototype.__abs__->operator()(this); }
	VirtualMachine::the().interpreter().raise_exception(
		"TypeError: bad operand type for abs(): '{}'", m_type_prototype.__name__);
	return nullptr;
}

PyObject *PyObject::neg() const
{
	if (m_type_prototype.__neg__.has_value()) { return m_type_prototype.__neg__->operator()(this); }
	VirtualMachine::the().interpreter().raise_exception(
		"bad operand type for unary -: '{}'", m_type_prototype.__name__);
	return nullptr;
}

PyObject *PyObject::pos() const
{
	if (m_type_prototype.__pos__.has_value()) { return m_type_prototype.__pos__->operator()(this); }
	VirtualMachine::the().interpreter().raise_exception(
		"bad operand type for unary +: '{}'", m_type_prototype.__name__);
	return nullptr;
}

PyObject *PyObject::invert() const
{
	if (m_type_prototype.__invert__.has_value()) {
		return m_type_prototype.__invert__->operator()(this);
	}
	VirtualMachine::the().interpreter().raise_exception(
		"bad operand type for unary ~: '{}'", m_type_prototype.__name__);
	return nullptr;
}


PyObject *PyObject::bool_() const
{
	ASSERT(m_type_prototype.__bool__.has_value())
	return m_type_prototype.__bool__->operator()(this);
}


PyObject *PyObject::len() const
{
	if (m_type_prototype.__len__.has_value()) { return m_type_prototype.__len__->operator()(this); }

	VirtualMachine::the().interpreter().raise_exception(
		fmt::format("TypeError: object of type '{}' has no len()", type()->name()));
	return nullptr;
}

PyObject *PyObject::iter() const
{
	if (m_type_prototype.__iter__.has_value()) {
		return m_type_prototype.__iter__->operator()(this);
	}

	VirtualMachine::the().interpreter().raise_exception(
		fmt::format("TypeError: '{}' object is not iterable", type()->name()));
	return nullptr;
}

PyObject *PyObject::next()
{
	if (m_type_prototype.__next__.has_value()) {
		return m_type_prototype.__next__->operator()(this);
	}

	VirtualMachine::the().interpreter().raise_exception(
		fmt::format("TypeError: '{}' object is not an iterator", type()->name()));
	return nullptr;
}


PyObject *PyObject::new_(PyTuple *args, PyDict *kwargs) const
{
	if (m_type_prototype.__new__.has_value()) {
		return m_type_prototype.__new__->operator()(type(), args, kwargs);
	}
	return nullptr;
}

std::optional<int32_t> PyObject::init(PyTuple *args, PyDict *kwargs)
{
	if (auto it = m_attributes.find("__init__"); it != m_attributes.end()) {
		PyObject *obj = it->second;
		if (obj->call(args, kwargs)) {
			return 0;
		} else {
			// maybe?
			return 1;
		}
	} else if (m_type_prototype.__init__.has_value()) {
		return m_type_prototype.__init__->operator()(this, args, kwargs);
	}
	return {};
}

PyObject *PyObject::__eq__(const PyObject *other) const
{
	return this == other ? py_true() : py_false();
}

PyObject *PyObject::__bool__() const { return py_true(); }

size_t PyObject::__hash__() const { return bit_cast<size_t>(this) >> 4; }

bool PyObject::is_callable() const { return m_type_prototype.__call__.has_value(); }

const std::string &PyObject::name() const { return m_type_prototype.__name__; }