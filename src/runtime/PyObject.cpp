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
					   [](PyObject *obj) -> size_t {
						   auto val = obj->hash();
						   ASSERT(val.is_ok())
						   ASSERT(val.unwrap_as<PyInteger>())
						   return val.unwrap_as<PyInteger>()->as_size_t();
					   } },
			value);
	return result;
}


bool ValueEqual::operator()(const Value &lhs_value, const Value &rhs_value) const
{
	const auto result =
		std::visit(overloaded{ [](PyObject *const lhs, PyObject *const rhs) {
								  auto r = lhs->richcompare(rhs, RichCompare::Py_EQ);
								  ASSERT(r.is_ok())
								  return r.template unwrap_as<PyObject>() == py_true();
							  },
					   [](PyObject *const lhs, const auto &rhs) { return lhs == rhs; },
					   [](const auto &lhs, PyObject *const rhs) { return lhs == rhs; },
					   [](const auto &lhs, const auto &rhs) { return lhs == rhs; } },
			lhs_value,
			rhs_value);
	return result;
}


template<> PyResult PyObject::from(PyObject *const &value)
{
	ASSERT(value)
	return PyResult::Ok(value);
}

template<> PyResult PyObject::from(const Number &value) { return PyNumber::create(value); }

template<> PyResult PyObject::from(const String &value) { return PyString::create(value.s); }

template<> PyResult PyObject::from(const Bytes &value) { return PyBytes::create(value); }

template<> PyResult PyObject::from(const Ellipsis &) { return PyResult::Ok(py_ellipsis()); }

template<> PyResult PyObject::from(const NameConstant &value)
{
	if (std::holds_alternative<NoneType>(value.value)) {
		return PyResult::Ok(py_none());
	} else {
		const bool bool_value = std::get<bool>(value.value);
		return PyResult::Ok(bool_value ? py_true() : py_false());
	}
}

template<> PyResult PyObject::from(const Value &value)
{
	return std::visit([](const auto &v) { return PyObject::from(v); }, value);
}

PyObject::PyObject(const TypePrototype &type) : Cell(), m_type_prototype(type) {}

void PyObject::visit_graph(Visitor &visitor)
{
	visitor.visit(*this);
	if (m_attributes) { visitor.visit(*m_attributes); }
}

PyResult PyObject::__repr__() const
{
	return PyString::create(fmt::format("<object at {}>", static_cast<const void *>(this)));
}

PyResult PyObject::richcompare(const PyObject *other, RichCompare op) const
{
	constexpr std::array opstr{ "<", "<=", "==", "!=", ">", ">=" };

	switch (op) {
	case RichCompare::Py_EQ: {
		if (auto result = eq(other);
			result.is_ok() && (result.unwrap_as<PyObject>() != not_implemented())) {
			return result;
		} else if (auto result = other->eq(this);
				   result.is_ok() && (result.unwrap_as<PyObject>() != not_implemented())) {
			return result;
		}
	} break;
	case RichCompare::Py_GE: {
		if (auto result = ge(other);
			result.is_ok() && (result.unwrap_as<PyObject>() != not_implemented())) {
			return result;
		} else if (auto result = other->le(this);
				   result.is_ok() && (result.unwrap_as<PyObject>() != not_implemented())) {
			return result;
		}
	} break;
	case RichCompare::Py_GT: {
		if (auto result = gt(other);
			result.is_ok() && (result.unwrap_as<PyObject>() != not_implemented())) {
			return result;
		} else if (auto result = other->lt(this);
				   result.is_ok() && (result.unwrap_as<PyObject>() != not_implemented())) {
			return result;
		}
	} break;
	case RichCompare::Py_LE: {
		if (auto result = le(other);
			result.is_ok() && (result.unwrap_as<PyObject>() != not_implemented())) {
			return result;
		} else if (auto result = other->ge(this);
				   result.is_ok() && (result.unwrap_as<PyObject>() != not_implemented())) {
			return result;
		}
	} break;
	case RichCompare::Py_LT: {
		if (auto result = lt(other);
			result.is_ok() && (result.unwrap_as<PyObject>() != not_implemented())) {
			return result;
		} else if (auto result = other->gt(this);
				   result.is_ok() && (result.unwrap_as<PyObject>() != not_implemented())) {
			return result;
		}
	} break;
	case RichCompare::Py_NE: {
		if (auto result = ne(other);
			result.is_ok() && (result.unwrap_as<PyObject>() != not_implemented())) {
			return result;
		} else if (auto result = other->ne(this);
				   result.is_ok() && (result.unwrap_as<PyObject>() != not_implemented())) {
			return result;
		}
	} break;
	}

	switch (op) {
	case RichCompare::Py_EQ: {
		return PyResult::Ok(this == other ? py_true() : py_false());
	} break;
	case RichCompare::Py_NE: {
		return PyResult::Ok(this != other ? py_true() : py_false());
	} break;
	default: {
		// op not supported
		return PyResult::Err(type_error("'{}' not supported between instances of '{}' and '{}'",
			opstr[static_cast<size_t>(op)],
			m_type_prototype.__name__,
			other->m_type_prototype.__name__));
	}
	}
}

namespace {
template<typename SlotFunctionType, typename... Args>
PyResult call_slot(const std::variant<SlotFunctionType, PyObject *> &slot, Args &&... args_)
{
	if (std::holds_alternative<SlotFunctionType>(slot)) {
		auto result = std::get<SlotFunctionType>(slot)(std::forward<Args>(args_)...);
		using ResultType = std::remove_pointer_t<decltype(result)>;

		if constexpr (std::is_same_v<PyResult, ResultType>) {
			return result;
		} else if constexpr (std::is_same_v<std::optional<int>, ResultType>) {
			ASSERT(result.has_value())
			return PyInteger::create(static_cast<int64_t>(*result));
		} else if constexpr (std::is_same_v<bool, ResultType>) {
			return PyResult::Ok(result ? py_true() : py_false());
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
		auto args = PyTuple::create(
			const_cast<PyObject *>(static_cast<const PyObject *>(std::forward<Args>(args_)))...);
		if (args.is_err()) { return args; }
		PyDict *kwargs = nullptr;
		return std::get<PyObject *>(slot)->call(args.template unwrap_as<PyTuple>(), kwargs);
	} else {
		TODO();
	}
}

template<typename SlotFunctionType>
PyResult call_slot(const std::variant<SlotFunctionType, PyObject *> &slot,
	PyObject *self,
	PyTuple *args,
	PyDict *kwargs)
{
	if (std::holds_alternative<SlotFunctionType>(slot)) {
		auto result = std::get<SlotFunctionType>(slot)(self, args, kwargs);
		using ResultType = std::remove_pointer_t<decltype(result)>;

		if constexpr (std::is_same_v<PyResult, ResultType>) {
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
		auto args_ = PyTuple::create(new_args);
		if (args_.is_err()) return args_;
		return std::get<PyObject *>(slot)->call(args_.unwrap_as<PyTuple>(), kwargs);
	} else {
		TODO();
	}
}

}// namespace

PyResult PyObject::eq(const PyObject *other) const
{
	if (m_type_prototype.__eq__.has_value()) {
		return call_slot(*m_type_prototype.__eq__, this, other);
	}
	return PyResult::Ok(not_implemented());
}

PyResult PyObject::ge(const PyObject *other) const
{
	if (m_type_prototype.__eq__.has_value()) {
		return call_slot(*m_type_prototype.__ge__, this, other);
	}
	return PyResult::Ok(not_implemented());
}

PyResult PyObject::gt(const PyObject *other) const
{
	if (m_type_prototype.__gt__.has_value()) {
		return call_slot(*m_type_prototype.__gt__, this, other);
	}
	return PyResult::Ok(not_implemented());
}

PyResult PyObject::le(const PyObject *other) const
{
	if (m_type_prototype.__le__.has_value()) {
		return call_slot(*m_type_prototype.__le__, this, other);
	}
	return PyResult::Ok(not_implemented());
}

PyResult PyObject::lt(const PyObject *other) const
{
	if (m_type_prototype.__lt__.has_value()) {
		return call_slot(*m_type_prototype.__lt__, this, other);
	}
	return PyResult::Ok(not_implemented());
}

PyResult PyObject::ne(const PyObject *other) const
{
	if (m_type_prototype.__ne__.has_value()) {
		return call_slot(*m_type_prototype.__ne__, this, other);
	}
	return PyResult::Ok(not_implemented());
}

PyResult PyObject::getattribute(PyObject *attribute) const
{
	if (m_type_prototype.__getattribute__.has_value()) {
		return call_slot(*m_type_prototype.__getattribute__, this, attribute);
	}
	TODO();
}

PyResult PyObject::setattribute(PyObject *attribute, PyObject *value)
{
	if (!as<PyString>(attribute)) {
		return PyResult::Err(
			type_error("attribute name must be string, not '{}'", attribute->type()->to_string()));
	}

	if (auto descriptor_ = type()->lookup(attribute); descriptor_.is_ok()) {
		auto *descriptor = descriptor_.unwrap_as<PyObject>();
		const auto &descriptor_set = descriptor->type()->underlying_type().__set__;
		if (descriptor_set.has_value()) {
			TODO();
			return call_slot(*descriptor_set, this, attribute, value)
				.and_then<PyObject>([](auto *obj) {
					return PyResult::Ok(obj == py_true() ? py_true() : py_false());
				});
		}
	}

	if (!m_attributes) {
		if (auto dict = PyDict::create(); dict.is_ok()) {
			m_attributes = dict.unwrap_as<PyDict>();
		} else {
			return dict;
		}
	}

	m_attributes->insert(attribute, value);

	return PyResult::Ok(py_none());
}

PyResult PyObject::repr() const
{
	if (m_type_prototype.__repr__.has_value()) {
		return call_slot(*m_type_prototype.__repr__, this);
	}
	TODO();
}

PyResult PyObject::hash() const
{
	if (m_type_prototype.__hash__.has_value()) {
		auto result_ = call_slot(*m_type_prototype.__hash__, this);
		if (result_.is_err()) return result_;
		auto *result = result_.unwrap_as<PyObject>();
		if (auto *result_int = as<PyInteger>(result)) { return result_; }
		return PyResult::Err(type_error(" __hash__ method should return an integer"));
	} else {
		return __hash__();
	}
}

PyResult PyObject::call(PyTuple *args, PyDict *kwargs)
{
	if (m_type_prototype.__call__.has_value()) {
		return call_slot(*m_type_prototype.__call__, this, args, kwargs);
	}
	return PyResult::Err(type_error("'{}' object is not callable", m_type_prototype.__name__));
}


PyResult PyObject::add(const PyObject *other) const
{
	if (m_type_prototype.__add__.has_value()) {
		return call_slot(*m_type_prototype.__add__, this, other);
	} else if (other->m_type_prototype.__add__.has_value()) {
		return call_slot(*other->m_type_prototype.__add__, other, this);
	}
	return PyResult::Err(type_error("unsupported operand type(s) for +: \'{}\' and \'{}\'",
		m_type_prototype.__name__,
		other->m_type_prototype.__name__));
}

PyResult PyObject::subtract(const PyObject *other) const
{
	if (m_type_prototype.__sub__.has_value()) {
		return call_slot(*m_type_prototype.__sub__, this, other);
	}
	return PyResult::Err(type_error("unsupported operand type(s) for -: \'{}\' and \'{}\'",
		m_type_prototype.__name__,
		other->m_type_prototype.__name__));
}

PyResult PyObject::multiply(const PyObject *other) const
{
	if (m_type_prototype.__mul__.has_value()) {
		return call_slot(*m_type_prototype.__mul__, this, other);
	} else if (other->m_type_prototype.__mul__.has_value()) {
		return call_slot(*other->m_type_prototype.__mul__, other, this);
	}
	return PyResult::Err(type_error("unsupported operand type(s) for *: \'{}\' and \'{}\'",
		m_type_prototype.__name__,
		other->m_type_prototype.__name__));
}

PyResult PyObject::exp(const PyObject *other) const
{
	if (m_type_prototype.__exp__.has_value()) {
		return call_slot(*m_type_prototype.__exp__, this, other);
	}
	return PyResult::Err(type_error("unsupported operand type(s) for **: \'{}\' and \'{}\'",
		m_type_prototype.__name__,
		other->m_type_prototype.__name__));
}

PyResult PyObject::lshift(const PyObject *other) const
{
	if (m_type_prototype.__lshift__.has_value()) {
		return call_slot(*m_type_prototype.__lshift__, this, other);
	}
	return PyResult::Err(type_error("unsupported operand type(s) for <<: \'{}\' and \'{}\'",
		m_type_prototype.__name__,
		other->m_type_prototype.__name__));
}

PyResult PyObject::modulo(const PyObject *other) const
{
	if (m_type_prototype.__mod__.has_value()) {
		return call_slot(*m_type_prototype.__mod__, this, other);
	}
	return PyResult::Err(type_error("unsupported operand type(s) for %: \'{}\' and \'{}\'",
		m_type_prototype.__name__,
		other->m_type_prototype.__name__));
}

PyResult PyObject::abs() const
{
	if (m_type_prototype.__abs__.has_value()) { return call_slot(*m_type_prototype.__abs__, this); }
	return PyResult::Err(type_error("bad operand type for abs(): '{}'", m_type_prototype.__name__));
}

PyResult PyObject::neg() const
{
	if (m_type_prototype.__neg__.has_value()) { return call_slot(*m_type_prototype.__neg__, this); }
	return PyResult::Err(
		type_error("bad operand type for unary -: '{}'", m_type_prototype.__name__));
}

PyResult PyObject::pos() const
{
	if (m_type_prototype.__pos__.has_value()) { return call_slot(*m_type_prototype.__pos__, this); }
	return PyResult::Err(
		type_error("bad operand type for unary +: '{}'", m_type_prototype.__name__));
}

PyResult PyObject::invert() const
{
	if (m_type_prototype.__invert__.has_value()) {
		return call_slot(*m_type_prototype.__invert__, this);
	}
	return PyResult::Err(
		type_error("bad operand type for unary ~: '{}'", m_type_prototype.__name__));
}


PyResult PyObject::bool_() const
{
	ASSERT(m_type_prototype.__bool__.has_value())
	return call_slot(*m_type_prototype.__bool__, this);
}


PyResult PyObject::len() const
{
	if (m_type_prototype.__len__.has_value()) { return call_slot(*m_type_prototype.__len__, this); }

	return PyResult::Err(type_error("object of type '{}' has no len()", type()->name()));
}

PyResult PyObject::iter() const
{
	if (m_type_prototype.__iter__.has_value()) {
		return call_slot(*m_type_prototype.__iter__, this);
	}

	return PyResult::Err(type_error("'{}' object is not iterable", type()->name()));
}

PyResult PyObject::next()
{
	if (m_type_prototype.__next__.has_value()) {
		return call_slot(*m_type_prototype.__next__, this);
	}

	return PyResult::Err(type_error("'{}' object is not an iterator", type()->name()));
}

PyResult PyObject::get(PyObject *instance, PyObject *owner) const
{
	if (m_type_prototype.__get__.has_value()) {
		return call_slot(*m_type_prototype.__get__, this, instance, owner);
	}
	TODO();
}

PyResult PyObject::new_(PyTuple *args, PyDict *kwargs) const
{
	if (!as<PyType>(this)) {
		// FIXME: should be SystemError
		return PyResult::Err(type_error("__new__() called with non-type 'self'"));
	}
	if (!args || args->size() < 1) {
		return PyResult::Err(type_error("object.__new__(): not enough arguments"));
	}
	auto maybe_type_ = PyObject::from(args->elements()[0]);
	if (maybe_type_.is_err()) return maybe_type_;
	auto *maybe_type = maybe_type_.unwrap_as<PyObject>();
	if (!as<PyType>(maybe_type)) {
		return PyResult::Err(type_error(
			"object.__new__(X): X is not a type object ({})", maybe_type->type()->name()));
	}
	auto *type = as<PyType>(maybe_type);
	// TODO: check type is a subtype of self (which we know here is `const PyType*`)
	//       otherwise raise TypeError("{}.__new__({}): {} is not a subtype of {}")

	// pop out type from args
	std::vector<Value> new_args;
	new_args.resize(args->size() - 1);
	new_args.insert(new_args.end(), args->elements().begin() + 1, args->elements().end());
	auto args_ = PyTuple::create(new_args);
	if (args_.is_err()) return args_;
	args = args_.unwrap_as<PyTuple>();

	if (m_type_prototype.__new__.has_value()) {
		return call_slot(*m_type_prototype.__new__, type, args, kwargs);
	}
	TODO();
}

std::optional<int32_t> PyObject::init(PyTuple *args, PyDict *kwargs)
{
	if (m_type_prototype.__init__.has_value()) {
		auto result_ = call_slot(*m_type_prototype.__init__, this, args, kwargs);
		ASSERT(result_.is_ok())
		if (std::holds_alternative<PyObject *>(result_.unwrap())) {
			auto *result = result_.unwrap_as<PyObject>();
			if (result == py_none()) {
				return 0;
			} else if (as<PyInteger>(result)) {
				return as<PyInteger>(result)->as_i64();
			}
		} else {
			const auto &result = result_.unwrap();
			if (std::holds_alternative<NameConstant>(result)) {
				if (std::holds_alternative<NoneType>(std::get<NameConstant>(result).value)) {
					return 0;
				}
			}
		}
		// should return type_error("__init__() should return None, not '{}'", result_type)
		TODO();
	}
	return {};
}

PyResult PyObject::__eq__(const PyObject *other) const
{
	return PyResult::Ok(this == other ? py_true() : py_false());
}

PyResult PyObject::__getattribute__(PyObject *attribute) const
{
	if (!as<PyString>(attribute)) {
		return PyResult::Err(
			type_error("attribute name must be a string, not {}", attribute->type()->to_string()));
	}

	auto *name = as<PyString>(attribute);

	auto descriptor_ = type()->lookup(name);
	if (descriptor_.is_err()) return descriptor_;
	auto *descriptor = descriptor_.unwrap_as<PyObject>();

	bool descriptor_has_get = false;
	if (descriptor) {
		const auto &descriptor_get = descriptor->type()->underlying_type().__get__;
		if (descriptor_get.has_value()) { descriptor_has_get = true; }
		if (descriptor_get.has_value() && descriptor_is_data(descriptor->type())) {
			return descriptor->get(const_cast<PyObject *>(this), type());
		}
	}

	if (m_attributes) {
		const auto &dict = m_attributes->map();
		if (auto it = dict.find(name); it != dict.end()) { return PyObject::from(it->second); }
		// FIXME: we should abort here if PyDict returns an exception that is not an
		// AttributeError
	}

	if (descriptor_has_get) { return descriptor->get(const_cast<PyObject *>(this), type()); }

	if (descriptor) { return descriptor_; }

	return PyResult::Err(attribute_error(
		"'{}' object has no attribute '{}'", m_type_prototype.__name__, name->to_string()));
}

PyResult PyObject::get_attribute(PyObject *name) const
{
	if (!as<PyString>(name)) {
		return PyResult::Err(
			type_error("attribute name must be a string, not {}", name->type()->to_string()));
	}
	const auto &getattribute_ = type()->underlying_type().__getattribute__;
	if (getattribute_.has_value()) { return getattribute(name); }

	return PyResult::Err(
		attribute_error("'{}' object has no attribute '{}'", type()->name(), name->to_string()));
}

std::tuple<PyResult, LookupAttrResult> PyObject::lookup_attribute(PyObject *name) const
{
	auto result = [&]() -> std::tuple<PyResult, LookupAttrResult> {
		if (!as<PyString>(name)) {
			return { PyResult::Err(type_error(
						 "attribute name must be a string, not {}", name->type()->to_string())),
				LookupAttrResult::NOT_FOUND };
		}

		const auto &getattribute_ = type()->underlying_type().__getattribute__;
		if (getattribute_.has_value()
			&& get_address(*getattribute_)
				   != get_address(*object()->underlying_type().__getattribute__)) {
			return { get_attribute(name), LookupAttrResult::FOUND };
		}
		// TODO: check tp_getattr? This field is deprecated in [c]python so maybe should not
		// even bother to implement it here?
		else if (getattribute_.has_value()) {
			return { call_slot(*getattribute_, this, name), LookupAttrResult::FOUND };
		} else {
			return { PyResult::Ok(py_none()), LookupAttrResult::NOT_FOUND };
		}
	}();

	if (std::get<0>(result).unwrap_err()->type() == AttributeError::static_type()) {
		return { PyResult::Ok(py_none()), LookupAttrResult::NOT_FOUND };
	} else {
		return result;
	}
}


PyResult PyObject::get_method(PyObject *name) const
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

	auto descriptor_ = type()->lookup(name);
	if (descriptor_.is_err()) return descriptor_;
	auto *descriptor = descriptor_.unwrap_as<PyObject>();

	if (descriptor) {
		if (is_method_descriptor(descriptor->type())) {
			method_found = true;
		} else {
			if (descriptor->type()->underlying_type().__get__.has_value()) {
				return descriptor->get(const_cast<PyObject *>(this), type());
			}
		}
	}

	if (m_attributes) {
		const auto &dict = m_attributes->map();
		if (auto it = dict.find(name); it != dict.end()) { return PyObject::from(it->second); }
	}

	if (method_found) {
		auto result = descriptor->get(const_cast<PyObject *>(this), type());
		ASSERT(result.is_ok())
		return result;
	}

	return PyResult::Err(attribute_error(
		"'{}' object has no attribute '{}'", m_type_prototype.__name__, name->to_string()));
}

PyResult PyObject::__setattribute__(PyObject *attribute, PyObject *value)
{
	if (as<PyString>(attribute)) {
		if (!m_attributes) {
			auto d = PyDict::create();
			if (d.is_err()) return d;
			m_attributes = d.unwrap_as<PyDict>();
		}
		m_attributes->insert(attribute, value);
	}
	TODO();
}

PyResult PyObject::__bool__() const { return PyResult::Ok(py_true()); }

PyResult PyObject::__hash__() const { return PyInteger::create(bit_cast<size_t>(this) >> 4); }

bool PyObject::is_callable() const { return m_type_prototype.__call__.has_value(); }

const std::string &PyObject::name() const { return m_type_prototype.__name__; }

PyResult PyObject::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
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
				return PyResult::Err(type_error(
					"object.__new__() takes exactly one argument (the type to instantiate)"));
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
				return PyResult::Err(type_error("object() takes no arguments"));
			}
		}
	}
	// FIXME: if custom allocators are ever added, should call the type's allocator here
	auto *obj = VirtualMachine::the().heap().allocate<CustomPyObject>(type);
	if (!obj) { return PyResult::Err(memory_error(sizeof(CustomPyObject))); }
	return PyResult::Ok(obj);
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
				TODO();
				// VirtualMachine::the().interpreter().raise_exception(type_error(
				// 	"object.__new__() takes exactly one argument (the type to instantiate)"));
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
				TODO();
				// VirtualMachine::the().interpreter().raise_exception(
				// 	type_error("object() takes no arguments"));
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