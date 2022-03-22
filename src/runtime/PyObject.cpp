#include "PyObject.hpp"

#include "AttributeError.hpp"
#include "CustomPyObject.hpp"
#include "PyBool.hpp"
#include "PyBoundMethod.hpp"
#include "PyBuiltInMethod.hpp"
#include "PyBytes.hpp"
#include "PyDict.hpp"
#include "PyEllipsis.hpp"
#include "PyFunction.hpp"
#include "PyInteger.hpp"
#include "PyMethodDescriptor.hpp"
#include "PyNone.hpp"
#include "PyNumber.hpp"
#include "PySlotWrapper.hpp"
#include "PyStaticMethod.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "PyType.hpp"
#include "StopIteration.hpp"
#include "TypeError.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

#include "executable/bytecode/instructions/FunctionCall.hpp"
#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"

using namespace py;

namespace {
bool is_method_descriptor(PyType *obj_type)
{
	return obj_type == method_wrapper() || obj_type == function();
}

bool descriptor_is_data(const PyObject *obj)
{
	// FIXME: temporary hack to get object.__new__ working, but requires __set__ to be implemented
	//        should be:
	//        obj->type()->underlying_type().__set__.has_value()
	return !as<PyStaticMethod>(obj) && !as<PySlotWrapper>(obj);
}
}// namespace

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

PyObject::PyObject(const TypePrototype &type) : Cell(), m_type_prototype(type) {}

void PyObject::visit_graph(Visitor &visitor)
{
	visitor.visit(*this);
	if (m_attributes) { visitor.visit(*m_attributes); }
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

namespace {
template<typename SlotFunctionType, typename... Args>
PyObject *call_slot(const std::variant<SlotFunctionType, PyObject *> &slot, Args &&... args_)
{
	if (std::holds_alternative<SlotFunctionType>(slot)) {
		auto result = std::get<SlotFunctionType>(slot)(std::forward<Args>(args_)...);
		using ResultType = std::remove_pointer_t<decltype(result)>;

		if constexpr (std::is_base_of_v<PyObject, ResultType>) {
			return result;
		} else if constexpr (std::is_same_v<std::optional<int>, ResultType>) {
			ASSERT(result.has_value())
			return PyInteger::create(static_cast<int64_t>(*result));
		} else if constexpr (std::is_same_v<bool, ResultType>) {
			return result ? py_true() : py_false();
		} else if constexpr (std::is_same_v<size_t, ResultType>) {
			return PyInteger::create(result);
		} else {
			[]<bool flag = false>() { static_assert(flag, "unsupported return type"); }
			();
		}
	} else if (std::holds_alternative<PyObject *>(slot)) {
		// FIXME: this const_cast is needed since in Python land there is no concept of
		//		  PyObject constness (right?). But for the internal calls handled above
		//		  which are resolved in the C++ runtime, we want to enforce constness
		//		  so we end up with the awkward line below. But how could we do better?
		auto *args = PyTuple::create(
			const_cast<PyObject *>(static_cast<const PyObject *>(std::forward<Args>(args_)))...);
		PyDict *kwargs = nullptr;
		return std::get<PyObject *>(slot)->call(args, kwargs);
	} else {
		TODO();
	}
}

template<typename SlotFunctionType>
PyObject *call_slot(const std::variant<SlotFunctionType, PyObject *> &slot,
	PyObject *self,
	PyTuple *args,
	PyDict *kwargs)
{
	if (std::holds_alternative<SlotFunctionType>(slot)) {
		auto result = std::get<SlotFunctionType>(slot)(self, args, kwargs);
		using ResultType = std::remove_pointer_t<decltype(result)>;

		if constexpr (std::is_base_of_v<PyObject, ResultType>) {
			return result;
		} else if constexpr (std::is_same_v<std::optional<int>, ResultType>) {
			ASSERT(result.has_value())
			return PyInteger::create(static_cast<int64_t>(*result));
		} else {
			[]<bool flag = false>() { static_assert(flag, "unsupported return type"); }
			();
		}
	} else if (std::holds_alternative<PyObject *>(slot)) {
		std::vector<Value> new_args;
		new_args.reserve(args->size() + 1);
		new_args.push_back(self);
		new_args.insert(new_args.end(), args->elements().begin(), args->elements().end());
		auto *args_ = PyTuple::create(new_args);
		return std::get<PyObject *>(slot)->call(args_, kwargs);
	} else {
		TODO();
	}
}

}// namespace

PyObject::PyResult PyObject::eq(const PyObject *other) const
{
	if (m_type_prototype.__eq__.has_value()) {
		return call_slot(*m_type_prototype.__eq__, this, other);
	}
	return NotImplemented_{};
}

PyObject::PyResult PyObject::ge(const PyObject *other) const
{
	if (m_type_prototype.__eq__.has_value()) {
		return call_slot(*m_type_prototype.__ge__, this, other);
	}
	return NotImplemented_{};
}

PyObject::PyResult PyObject::gt(const PyObject *other) const
{
	if (m_type_prototype.__gt__.has_value()) {
		return call_slot(*m_type_prototype.__gt__, this, other);
	}
	return NotImplemented_{};
}

PyObject::PyResult PyObject::le(const PyObject *other) const
{
	if (m_type_prototype.__le__.has_value()) {
		return call_slot(*m_type_prototype.__le__, this, other);
	}
	return NotImplemented_{};
}

PyObject::PyResult PyObject::lt(const PyObject *other) const
{
	if (m_type_prototype.__lt__.has_value()) {
		return call_slot(*m_type_prototype.__lt__, this, other);
	}
	return NotImplemented_{};
}

PyObject::PyResult PyObject::ne(const PyObject *other) const
{
	if (m_type_prototype.__ne__.has_value()) {
		return call_slot(*m_type_prototype.__ne__, this, other);
	}
	return NotImplemented_{};
}

PyObject *PyObject::getattribute(PyObject *attribute) const
{
	if (m_type_prototype.__getattribute__.has_value()) {
		return call_slot(*m_type_prototype.__getattribute__, this, attribute);
	}
	TODO();
}

bool PyObject::setattribute(PyObject *attribute, PyObject *value)
{
	if (!as<PyString>(attribute)) {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("attribute name must be string, not '{}'", attribute->type()->to_string()));
		return false;
	}

	auto *descriptor = type()->lookup(attribute);

	if (descriptor) {
		const auto &descriptor_set = descriptor->type()->underlying_type().__set__;
		if (descriptor_set.has_value()) {
			TODO();
			return call_slot(*descriptor_set, this, attribute, value) == py_true();
		}
	}

	if (!m_attributes) { m_attributes = PyDict::create(); }

	m_attributes->insert(attribute, value);

	return true;
}

PyObject *PyObject::repr() const
{
	if (m_type_prototype.__repr__.has_value()) {
		return call_slot(*m_type_prototype.__repr__, this);
	}
	TODO();
}

size_t PyObject::hash() const
{
	if (m_type_prototype.__hash__.has_value()) {
		auto *result = call_slot(*m_type_prototype.__hash__, this);
		if (auto *result_int = as<PyInteger>(result)) { return result_int->as_size_t(); }
		VirtualMachine::the().interpreter().raise_exception(
			type_error(" __hash__ method should return an integer"));
		return 0;
	} else {
		return bit_cast<size_t>(this);
	}
}

PyObject *PyObject::call(PyTuple *args, PyDict *kwargs)
{
	if (m_type_prototype.__call__.has_value()) {
		return call_slot(*m_type_prototype.__call__, this, args, kwargs);
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("'{}' object is not callable", m_type_prototype.__name__));
	return nullptr;
}


PyObject *PyObject::add(const PyObject *other) const
{
	if (m_type_prototype.__add__.has_value()) {
		return call_slot(*m_type_prototype.__add__, this, other);
	} else if (other->m_type_prototype.__add__.has_value()) {
		return call_slot(*other->m_type_prototype.__add__, other, this);
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("unsupported operand type(s) for +: \'{}\' and \'{}\'",
			m_type_prototype.__name__,
			other->m_type_prototype.__name__));
	return nullptr;
}

PyObject *PyObject::subtract(const PyObject *other) const
{
	if (m_type_prototype.__sub__.has_value()) {
		return call_slot(*m_type_prototype.__sub__, this, other);
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("unsupported operand type(s) for -: \'{}\' and \'{}\'",
			m_type_prototype.__name__,
			other->m_type_prototype.__name__));
	return nullptr;
}

PyObject *PyObject::multiply(const PyObject *other) const
{
	if (m_type_prototype.__mul__.has_value()) {
		return call_slot(*m_type_prototype.__mul__, this, other);
	} else if (other->m_type_prototype.__mul__.has_value()) {
		return call_slot(*other->m_type_prototype.__mul__, other, this);
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("unsupported operand type(s) for *: \'{}\' and \'{}\'",
			m_type_prototype.__name__,
			other->m_type_prototype.__name__));
	return nullptr;
}

PyObject *PyObject::exp(const PyObject *other) const
{
	if (m_type_prototype.__exp__.has_value()) {
		return call_slot(*m_type_prototype.__exp__, this, other);
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("unsupported operand type(s) for **: \'{}\' and \'{}\'",
			m_type_prototype.__name__,
			other->m_type_prototype.__name__));
	return nullptr;
}

PyObject *PyObject::lshift(const PyObject *other) const
{
	if (m_type_prototype.__lshift__.has_value()) {
		return call_slot(*m_type_prototype.__lshift__, this, other);
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("unsupported operand type(s) for <<: \'{}\' and \'{}\'",
			m_type_prototype.__name__,
			other->m_type_prototype.__name__));
	return nullptr;
}

PyObject *PyObject::modulo(const PyObject *other) const
{
	if (m_type_prototype.__mod__.has_value()) {
		return call_slot(*m_type_prototype.__mod__, this, other);
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("unsupported operand type(s) for %: \'{}\' and \'{}\'",
			m_type_prototype.__name__,
			other->m_type_prototype.__name__));
	return nullptr;
}

PyObject *PyObject::abs() const
{
	if (m_type_prototype.__abs__.has_value()) { return call_slot(*m_type_prototype.__abs__, this); }
	VirtualMachine::the().interpreter().raise_exception(
		type_error("bad operand type for abs(): '{}'", m_type_prototype.__name__));
	return nullptr;
}

PyObject *PyObject::neg() const
{
	if (m_type_prototype.__neg__.has_value()) { return call_slot(*m_type_prototype.__neg__, this); }
	VirtualMachine::the().interpreter().raise_exception(
		type_error("bad operand type for unary -: '{}'", m_type_prototype.__name__));
	return nullptr;
}

PyObject *PyObject::pos() const
{
	if (m_type_prototype.__pos__.has_value()) { return call_slot(*m_type_prototype.__pos__, this); }
	VirtualMachine::the().interpreter().raise_exception(
		type_error("bad operand type for unary +: '{}'", m_type_prototype.__name__));
	return nullptr;
}

PyObject *PyObject::invert() const
{
	if (m_type_prototype.__invert__.has_value()) {
		return call_slot(*m_type_prototype.__invert__, this);
	}
	VirtualMachine::the().interpreter().raise_exception(
		type_error("bad operand type for unary ~: '{}'", m_type_prototype.__name__));
	return nullptr;
}


PyObject *PyObject::bool_() const
{
	ASSERT(m_type_prototype.__bool__.has_value())
	return call_slot(*m_type_prototype.__bool__, this);
}


PyObject *PyObject::len() const
{
	if (m_type_prototype.__len__.has_value()) { return call_slot(*m_type_prototype.__len__, this); }

	VirtualMachine::the().interpreter().raise_exception(
		type_error("object of type '{}' has no len()", type()->name()));
	return nullptr;
}

PyObject *PyObject::iter() const
{
	if (m_type_prototype.__iter__.has_value()) {
		return call_slot(*m_type_prototype.__iter__, this);
	}

	VirtualMachine::the().interpreter().raise_exception(
		type_error("'{}' object is not iterable", type()->name()));
	return nullptr;
}

PyObject *PyObject::next()
{
	if (m_type_prototype.__next__.has_value()) {
		return call_slot(*m_type_prototype.__next__, this);
	}

	VirtualMachine::the().interpreter().raise_exception(
		type_error("'{}' object is not an iterator", type()->name()));
	return nullptr;
}

PyObject *PyObject::get(PyObject *instance, PyObject *owner) const
{
	if (m_type_prototype.__get__.has_value()) {
		return call_slot(*m_type_prototype.__get__, this, instance, owner);
	} else {
		TODO();
	}
	return nullptr;
}

PyObject *PyObject::new_(PyTuple *args, PyDict *kwargs) const
{
	if (!as<PyType>(this)) {
		// FIXME: should be SystemError
		VirtualMachine::the().interpreter().raise_exception(
			type_error("__new__() called with non-type 'self'"));
		return nullptr;
	}
	if (!args || args->size() < 1) {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("object.__new__(): not enough arguments"));
		return nullptr;
	}
	auto *maybe_type = PyObject::from(args->elements()[0]);
	if (!as<PyType>(maybe_type)) {
		VirtualMachine::the().interpreter().raise_exception(type_error(
			"object.__new__(X): X is not a type object ({})", maybe_type->type()->name()));
	}
	auto *type = as<PyType>(maybe_type);
	// TODO: check type is a subtype of self (which we know here is `const PyType*`)
	//       otherwise raise TypeError("{}.__new__({}): {} is not a subtype of {}")

	// pop out type from args
	std::vector<Value> new_args;
	new_args.resize(args->size() - 1);
	new_args.insert(new_args.end(), args->elements().begin() + 1, args->elements().end());
	args = PyTuple::create(new_args);

	if (m_type_prototype.__new__.has_value()) {
		return call_slot(*m_type_prototype.__new__, type, args, kwargs);
	}
	return nullptr;
}

std::optional<int32_t> PyObject::init(PyTuple *args, PyDict *kwargs)
{
	if (m_type_prototype.__init__.has_value()) {
		auto *result = call_slot(*m_type_prototype.__init__, this, args, kwargs);
		if (as<PyInteger>(result) || result == py_none()) { return 0; }
		TODO();
	}
	return {};
}

PyObject *PyObject::__eq__(const PyObject *other) const
{
	return this == other ? py_true() : py_false();
}

PyObject *PyObject::__getattribute__(PyObject *attribute) const
{
	if (!as<PyString>(attribute)) {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("attribute name must be a string, not {}", attribute->type()->to_string()));
		return nullptr;
	}

	auto *name = as<PyString>(attribute);

	auto *descriptor = type()->lookup(name);
	bool descriptor_has_get = false;
	if (descriptor) {
		const auto &descriptor_get = descriptor->type()->underlying_type().__get__;
		if (descriptor_get.has_value()) { descriptor_has_get = true; }
		if (descriptor_get.has_value() && descriptor_is_data(descriptor->type())) {
			auto *res = descriptor->get(const_cast<PyObject *>(this), type());
			if (!res) { TODO(); }
			return res;
		}
	}

	if (m_attributes) {
		const auto &dict = m_attributes->map();
		if (auto it = dict.find(name); it != dict.end()) { return PyObject::from(it->second); }
		// FIXME: we should abort here if PyDict returns an exception that is not an AttributeError
	}

	if (descriptor_has_get) { return descriptor->get(const_cast<PyObject *>(this), type()); }

	if (descriptor) { return descriptor; }

	VirtualMachine::the().interpreter().raise_exception(attribute_error(
		"'{}' object has no attribute '{}'", m_type_prototype.__name__, name->to_string()));
	return nullptr;
}

PyObject *PyObject::get_attribute(PyObject *name) const
{
	if (!as<PyString>(name)) {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("attribute name must be a string, not {}", name->type()->to_string()));
		return nullptr;
	}
	const auto &getattribute_ = type()->underlying_type().__getattribute__;
	if (getattribute_.has_value()) { return getattribute(name); }

	VirtualMachine::the().interpreter().raise_exception(
		attribute_error("'{}' object has no attribute '{}'", type()->name(), name->to_string()));
	return nullptr;
}

std::tuple<PyObject *, LookupAttrResult> PyObject::lookup_attribute(PyObject *name) const
{
	PyObject *result = nullptr;
	if (!as<PyString>(name)) {
		VirtualMachine::the().interpreter().raise_exception(
			type_error("attribute name must be a string, not {}", name->type()->to_string()));
		return { nullptr, LookupAttrResult::ERROR };
	}

	const auto &getattribute_ = type()->underlying_type().__getattribute__;
	if (getattribute_.has_value()
		&& get_address(*getattribute_)
			   != get_address(*object()->underlying_type().__getattribute__)) {
		result = get_attribute(name);
	}
	// TODO: check tp_getattr? This field is deprecated in [c]python so maybe should not even bother
	// to implement it here?
	else if (getattribute_.has_value()) {
		result = call_slot(*getattribute_, this, name);
	} else {
		return { nullptr, LookupAttrResult::NOT_FOUND };
	}

	// TODO: check if error occured
	if (result) {
		return { result, LookupAttrResult::FOUND };
	} else if (auto *exception = VirtualMachine::the()
									 .interpreter()
									 .execution_frame()
									 ->exception_info()
									 ->exception;
			   exception && exception->type() == AttributeError::static_type()) {
		VirtualMachine::the().interpreter().execution_frame()->set_exception(nullptr);
		VirtualMachine::the().interpreter().set_status(Interpreter::Status::OK);
		return { nullptr, LookupAttrResult::NOT_FOUND };
	} else {
		return { nullptr, LookupAttrResult::ERROR };
	}
}


PyObject *PyObject::get_method(PyObject *name) const
{
	bool method_found = false;

	{
		// check if the derived object uses the default __getattribute__
		const auto &getattribute_ = type()->underlying_type().__getattribute__;
		if (getattribute_.has_value()
			&& get_address(*getattribute_)
				   != get_address(*object()->underlying_type().__getattribute__)) {
			return get_attribute(name);
		}
	}

	auto *descriptor = type()->lookup(name);
	if (descriptor) {
		if (is_method_descriptor(descriptor->type())) {
			method_found = true;
		} else {
			if (descriptor->type()->underlying_type().__get__.has_value()) {
				auto result = descriptor->get(const_cast<PyObject *>(this), type());
				ASSERT(result)
				return result;
			}
		}
	}

	if (m_attributes) {
		const auto &dict = m_attributes->map();
		if (auto it = dict.find(name); it != dict.end()) { return PyObject::from(it->second); }
	}

	if (method_found) {
		auto result = descriptor->get(const_cast<PyObject *>(this), type());
		ASSERT(result)
		return result;
	}

	VirtualMachine::the().interpreter().raise_exception(attribute_error(
		"'{}' object has no attribute '{}'", m_type_prototype.__name__, name->to_string()));
	return nullptr;
}

PyObject *PyObject::__setattribute__(PyObject *attribute, PyObject *value)
{
	if (as<PyString>(attribute)) {
		if (!m_attributes) { m_attributes = PyDict::create(); }
		m_attributes->insert(attribute, value);
	} else {
		TODO();
	}
	return py_none();
}

PyObject *PyObject::__bool__() const { return py_true(); }

size_t PyObject::__hash__() const { return bit_cast<size_t>(this) >> 4; }

bool PyObject::is_callable() const { return m_type_prototype.__call__.has_value(); }

const std::string &PyObject::name() const { return m_type_prototype.__name__; }

PyObject *PyObject::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	if ((args && !args->elements().empty()) || (kwargs && !kwargs->map().empty())) {

		if (!type->underlying_type().__dict__->map().contains(String{ "__new__" })) {
			ASSERT(type->underlying_type().__new__)
			const auto new_fn = get_address(*type->underlying_type().__new__);
			ASSERT(new_fn)

			ASSERT(object()->underlying_type().__new__)
			const auto custom_new_fn = get_address(*object()->underlying_type().__new__);
			ASSERT(custom_new_fn)

			if (new_fn != custom_new_fn) {
				VirtualMachine::the().interpreter().raise_exception(type_error(
					"object.__new__() takes exactly one argument (the type to instantiate)"));
				return nullptr;
			}
		}

		if (!type->underlying_type().__dict__->map().contains(String{ "__init__" })) {
			ASSERT(type->underlying_type().__init__)
			const auto init_fn = get_address(*type->underlying_type().__init__);
			ASSERT(init_fn)

			ASSERT(object()->underlying_type().__init__)
			const auto custom_init_fn = get_address(*object()->underlying_type().__init__);
			ASSERT(object)
			if (init_fn == custom_init_fn) {
				VirtualMachine::the().interpreter().raise_exception(
					type_error("object() takes no arguments"));
				return nullptr;
			}
		}
	}
	// FIXME: if custom allocators are ever added, should call the type's allocator here
	return VirtualMachine::the().heap().allocate<CustomPyObject>(type);
}

std::optional<int32_t> PyObject::__init__(PyTuple *args, PyDict *kwargs)
{
	if ((args && !args->elements().empty()) || (kwargs && !kwargs->map().empty())) {
		if (!type()->underlying_type().__dict__->map().contains(String{ "__new__" })) {
			ASSERT(type()->underlying_type().__new__)
			const auto new_fn = get_address(*type()->underlying_type().__new__);
			ASSERT(new_fn)

			ASSERT(object()->underlying_type().__new__)
			const auto custom_new_fn = get_address(*object()->underlying_type().__new__);
			ASSERT(custom_new_fn)

			if (new_fn == custom_new_fn) {
				VirtualMachine::the().interpreter().raise_exception(type_error(
					"object.__new__() takes exactly one argument (the type to instantiate)"));
				return -1;
			}
		}

		if (!type()->underlying_type().__dict__->map().contains(String{ "__init__" })) {
			ASSERT(type()->underlying_type().__init__)
			const auto init_fn = get_address(*type()->underlying_type().__init__);
			ASSERT(init_fn)

			ASSERT(object()->underlying_type().__init__)
			const auto custom_init_fn = get_address(*object()->underlying_type().__init__);
			ASSERT(custom_init_fn)

			if (init_fn != custom_init_fn) {
				VirtualMachine::the().interpreter().raise_exception(
					type_error("object() takes no arguments"));
				return -1;
			}
		}
	}
	return 0;
}

std::string PyObject::to_string() const
{
	return fmt::format("PyObject at {}", static_cast<const void *>(this));
}

PyType *PyObject::type() const { return object(); }

namespace {

std::once_flag object_type_flag;

std::unique_ptr<TypePrototype> register_type() { return std::move(klass<PyObject>("object").type); }
}// namespace

std::unique_ptr<TypePrototype> PyObject::register_type()
{
	static std::unique_ptr<TypePrototype> type = nullptr;
	std::call_once(object_type_flag, []() { type = ::register_type(); });
	return std::move(type);
}