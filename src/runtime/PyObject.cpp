#include "PyObject.hpp"

#include "AttributeError.hpp"
#include "CustomPyObject.hpp"
#include "NotImplementedError.hpp"
#include "PyBool.hpp"
#include "PyBoundMethod.hpp"
#include "PyBuiltInMethod.hpp"
#include "PyBytes.hpp"
#include "PyDict.hpp"
#include "PyEllipsis.hpp"
#include "PyFunction.hpp"
#include "PyGenericAlias.hpp"
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
	return obj_type == method_wrapper() || obj_type == function()
		   || obj_type == classmethod_descriptor();
}

bool descriptor_is_data(const PyObject *obj)
{
	// FIXME: temporary hack to get object.__new__ working, but requires __set__ to be implemented
	//        should be:
	//        obj->type()->underlying_type().__set__.has_value()
	return !as<PyStaticMethod>(obj) && !as<PySlotWrapper>(obj);
}
}// namespace

void TypePrototype::visit_graph(::Cell::Visitor &visitor)
{
	if (__class__) { visitor.visit(*__class__); }
	if (__dict__) { visitor.visit(*__dict__); }
	if (__mro__) { visitor.visit(*__mro__); }
}

size_t ValueHash::operator()(const Value &value) const
{
	// TODO: put these hash functions somewhere global so they can be reused by the respective
	//       PyObject classes
	const auto result =
		std::visit(overloaded{ [](const Number &number) -> size_t {
								  if (std::holds_alternative<double>(number.value)) {
									  return std::hash<double>{}(std::get<double>(number.value));
								  } else {
									  if (std::get<int64_t>(number.value) == -1) return -2;
									  return std::get<int64_t>(number.value);
								  }
							  },
					   [](const String &s) -> size_t { return std::hash<std::string>{}(s.s); },
					   [](const Bytes &b) -> size_t { return ::bit_cast<size_t>(b.b.data()); },
					   [](const Ellipsis &) -> size_t { return ::bit_cast<size_t>(py_ellipsis()); },
					   [](const NameConstant &c) -> size_t {
						   if (std::holds_alternative<bool>(c.value)) {
							   return std::get<bool>(c.value) ? 0 : 1;
						   } else {
							   return bit_cast<size_t>(py_none()) >> 4;
						   }
					   },
					   [](PyObject *obj) -> size_t {
						   auto val = obj->hash();
						   ASSERT(val.is_ok())
						   return val.unwrap();
					   } },
			value);
	return result;
}

namespace {

template<typename SlotFunctionType,
	typename ResultType = typename SlotFunctionType::result_type,
	typename... Args>
ResultType call_slot(const std::variant<SlotFunctionType, PyObject *> &slot,
	Args &&...args_) requires std::is_same_v<typename ResultType::OkType, PyObject *>
{
	if (std::holds_alternative<SlotFunctionType>(slot)) {
		return std::get<SlotFunctionType>(slot)(std::forward<Args>(args_)...);
	} else if (std::holds_alternative<PyObject *>(slot)) {
		// FIXME: this const_cast is needed since in Python land there is no concept of
		//		  PyObject constness (right?). But for the internal calls handled above
		//		  which are resolved in the C++ runtime, we want to enforce constness
		//		  so we end up with the awkward line below. But how could we do better?
		auto args = PyTuple::create(
			const_cast<PyObject *>(static_cast<const PyObject *>(std::forward<Args>(args_)))...);
		if (args.is_err()) { return Err(args.unwrap_err()); }
		PyDict *kwargs = nullptr;
		return std::get<PyObject *>(slot)->call(args.unwrap(), kwargs);
	} else {
		TODO();
	}
}

template<typename SlotFunctionType,
	typename ResultType = typename SlotFunctionType::result_type,
	typename... Args>
ResultType call_slot(const std::variant<SlotFunctionType, PyObject *> &slot,
	std::string_view conversion_error_message,
	Args &&...args_) requires(!std::is_same_v<typename ResultType::OkType, PyObject *>)
{
	if (std::holds_alternative<SlotFunctionType>(slot)) {
		return std::get<SlotFunctionType>(slot)(std::forward<Args>(args_)...);
	} else if (std::holds_alternative<PyObject *>(slot)) {
		// FIXME: this const_cast is needed since in Python land there is no concept of
		//		  PyObject constness (right?). But for the internal calls handled above
		//		  which are resolved in the C++ runtime, we want to enforce constness
		//		  so we end up with the awkward line below. But how could we do better?
		auto args = PyTuple::create(
			const_cast<PyObject *>(static_cast<const PyObject *>(std::forward<Args>(args_)))...);
		if (args.is_err()) { return Err(args.unwrap_err()); }
		PyDict *kwargs = nullptr;
		if constexpr (std::is_same_v<typename ResultType::OkType, bool>) {
			auto result = std::get<PyObject *>(slot)->call(args.unwrap(), kwargs);
			if (result.is_err()) return Err(result.unwrap_err());
			if (!as<PyBool>(result.unwrap())) {
				return Err(type_error(std::string(conversion_error_message)));
			}
			return Ok(as<PyBool>(result.unwrap())->value());
		} else if constexpr (std::is_integral_v<typename ResultType::OkType>) {
			auto result = std::get<PyObject *>(slot)->call(args.unwrap(), kwargs);
			if (result.is_err()) return Err(result.unwrap_err());
			if (!as<PyInteger>(result.unwrap())) {
				return Err(type_error(std::string(conversion_error_message)));
			}
			return Ok(as<PyInteger>(result.unwrap())->as_i64());
		} else if constexpr (std::is_same_v<typename ResultType::OkType, std::monostate>) {
			auto result = std::get<PyObject *>(slot)->call(args.unwrap(), kwargs);
			if (result.is_err()) {
				return Err(result.unwrap_err());
			} else {
				return Ok(std::monostate{});
			}
		} else {
			[]<bool flag = false>() { static_assert(flag, "unsupported return type"); }
			();
		}
	} else {
		TODO();
	}
}

}// namespace


PyResult<size_t> PyMappingWrapper::len()
{
	if (!m_object->type_prototype().mapping_type_protocol.has_value()) { TODO(); }
	if (m_object->type_prototype().mapping_type_protocol->__len__.has_value()) {
		return call_slot(*m_object->type_prototype().mapping_type_protocol->__len__,
			"object cannot be interpreted as an integer",
			m_object);
	}

	return Err(type_error("object of type '{}' has no len()", m_object->type()->name()));
}

PyResult<PyObject *> PyMappingWrapper::getitem(PyObject *name)
{
	if (!m_object->type_prototype().mapping_type_protocol.has_value()) { TODO(); }
	if (m_object->type_prototype().mapping_type_protocol->__getitem__.has_value()) {
		return call_slot(
			*m_object->type_prototype().mapping_type_protocol->__getitem__, m_object, name);
	}

	return Err(type_error("object of type '{}' is not subscriptable", m_object->type()->name()));
}

PyResult<std::monostate> PyMappingWrapper::setitem(PyObject *name, PyObject *value)
{
	if (!m_object->type_prototype().mapping_type_protocol.has_value()) { TODO(); }
	if (m_object->type_prototype().mapping_type_protocol->__setitem__.has_value()) {
		return call_slot(*m_object->type_prototype().mapping_type_protocol->__setitem__,
			"",
			m_object,
			name,
			value);
	}

	return Err(type_error(
		"object of type '{}' does not support item assignment", m_object->type()->name()));
}

PyResult<std::monostate> PyMappingWrapper::delitem(PyObject *name)
{
	if (!m_object->type_prototype().mapping_type_protocol.has_value()) { TODO(); }
	if (m_object->type_prototype().mapping_type_protocol->__delitem__.has_value()) {
		return call_slot(
			*m_object->type_prototype().mapping_type_protocol->__delitem__, "", m_object, name);
	}

	return Err(
		type_error("object of type '{}' does not support item deletion", m_object->type()->name()));
}

PyResult<size_t> PySequenceWrapper::len()
{
	if (!m_object->type_prototype().sequence_type_protocol.has_value()) { TODO(); }
	if (m_object->type_prototype().sequence_type_protocol->__len__.has_value()) {
		return call_slot(*m_object->type_prototype().sequence_type_protocol->__len__,
			"object cannot be interpreted as an integer",
			m_object);
	}

	return Err(type_error("object of type '{}' has no len()", m_object->type()->name()));
}

PyResult<bool> PySequenceWrapper::contains(PyObject *value)
{
	if (!m_object->type_prototype().sequence_type_protocol.has_value()) { TODO(); }
	if (m_object->type_prototype().sequence_type_protocol->__contains__.has_value()) {
		return call_slot(*m_object->type_prototype().sequence_type_protocol->__contains__,
			"object cannot be interpreted as a bool",
			m_object,
			value);
	}

	return Err(type_error("object of type '{}' has no contains()", m_object->type()->name()));
}

PyResult<PyObject *> PySequenceWrapper::concat(PyObject *value)
{
	if (!m_object->type_prototype().sequence_type_protocol.has_value()) { TODO(); }
	if (m_object->type_prototype().sequence_type_protocol->__concat__.has_value()) {
		return call_slot(
			*m_object->type_prototype().sequence_type_protocol->__concat__, m_object, value);
	}

	return Err(type_error("unsupported operand type(s) for +: \'{}\' and \'{}\'",
		m_object->type()->name(),
		value->type()->name()));
}

template<> PyResult<PyObject *> PyObject::from(PyObject *const &value)
{
	ASSERT(value)
	return Ok(value);
}

template<> PyResult<PyObject *> PyObject::from(const Number &value)
{
	return PyNumber::create(value);
}

template<> PyResult<PyObject *> PyObject::from(const String &value)
{
	return PyString::create(value.s);
}

template<> PyResult<PyObject *> PyObject::from(const Bytes &value)
{
	return PyBytes::create(value);
}

template<> PyResult<PyObject *> PyObject::from(const Ellipsis &) { return Ok(py_ellipsis()); }

template<> PyResult<PyObject *> PyObject::from(const NameConstant &value)
{
	if (std::holds_alternative<NoneType>(value.value)) {
		return Ok(py_none());
	} else {
		const bool bool_value = std::get<bool>(value.value);
		return Ok(bool_value ? py_true() : py_false());
	}
}

template<> PyResult<PyObject *> PyObject::from(const Value &value)
{
	return std::visit([](const auto &v) { return PyObject::from(v); }, value);
}

PyObject::PyObject(const TypePrototype &type) : Cell(), m_type(type) {}

PyObject::PyObject(PyType *type) : Cell(), m_type(type) {}

const TypePrototype &PyObject::type_prototype() const
{
	if (std::holds_alternative<std::reference_wrapper<const TypePrototype>>(m_type)) {
		return std::get<std::reference_wrapper<const TypePrototype>>(m_type).get();
	} else {
		auto *type = std::get<PyType *>(m_type);
		ASSERT(type);
		return type->underlying_type();
	}
}

void PyObject::visit_graph(Visitor &visitor)
{
	if (m_attributes) { visitor.visit(*m_attributes); }
	if (std::holds_alternative<PyType *>(m_type)) {
		if (auto *t = std::get<PyType *>(m_type)) { visitor.visit(*t); }
	} else {
		const_cast<TypePrototype &>(
			std::get<std::reference_wrapper<const TypePrototype>>(m_type).get())
			.visit_graph(visitor);
	}
}

PyResult<PyMappingWrapper> PyObject::as_mapping()
{
	if (type_prototype().mapping_type_protocol.has_value()) {
		return Ok(PyMappingWrapper(this));
	} else {
		return Err(type_error("object of type {} is not mappable", type_prototype().__name__));
	}
}

PyResult<PySequenceWrapper> PyObject::as_sequence()
{
	if (type_prototype().sequence_type_protocol.has_value()) {
		return Ok(PySequenceWrapper(this));
	} else {
		return Err(type_error("object of type {} is not a sequence", type_prototype().__name__));
	}
}

PyResult<PyBufferProcs> PyObject::as_buffer()
{
	if (type_prototype().as_buffer.has_value()) {
		return Ok(*type_prototype().as_buffer);
	} else {
		return Err(
			type_error("a bytes-like object is required, not '{}'", type_prototype().__name__));
	}
}

PyResult<std::monostate> PyObject::get_buffer(PyBuffer &buffer, int flags)
{
	return as_buffer().and_then(
		[&buffer, flags, this](const PyBufferProcs &bc) -> PyResult<std::monostate> {
			(void)buffer;
			(void)flags;
			(void)this;
			(void)bc;
			return Err(not_implemented_error("get_buffer not implemented!"));
			// return bc.getbuffer(this, buffer, flags);
		});
}

PyResult<PyObject *> PyObject::__repr__() const
{
	return PyString::create(fmt::format(
		"<{} object at {}>", type_prototype().__name__, static_cast<const void *>(this)));
}

PyResult<PyObject *> PyObject::__str__()
{
	if (!type()->underlying_type().__repr__.has_value()) { return PyObject::__repr__(); }
	return repr();
}

PyResult<PyObject *> PyObject::richcompare(const PyObject *other, RichCompare op) const
{
	constexpr std::array opstr{ "<", "<=", "==", "!=", ">", ">=" };

	switch (op) {
	case RichCompare::Py_EQ: {
		if (auto result = eq(other); result.is_ok() && (result.unwrap() != not_implemented())) {
			return result;
		} else if (auto result = other->eq(this);
				   result.is_ok() && (result.unwrap() != not_implemented())) {
			return result;
		}
	} break;
	case RichCompare::Py_GE: {
		if (auto result = ge(other); result.is_ok() && (result.unwrap() != not_implemented())) {
			return result;
		} else if (auto result = other->le(this);
				   result.is_ok() && (result.unwrap() != not_implemented())) {
			return result;
		}
	} break;
	case RichCompare::Py_GT: {
		if (auto result = gt(other); result.is_ok() && (result.unwrap() != not_implemented())) {
			return result;
		} else if (auto result = other->lt(this);
				   result.is_ok() && (result.unwrap() != not_implemented())) {
			return result;
		}
	} break;
	case RichCompare::Py_LE: {
		if (auto result = le(other); result.is_ok() && (result.unwrap() != not_implemented())) {
			return result;
		} else if (auto result = other->ge(this);
				   result.is_ok() && (result.unwrap() != not_implemented())) {
			return result;
		}
	} break;
	case RichCompare::Py_LT: {
		if (auto result = lt(other); result.is_ok() && (result.unwrap() != not_implemented())) {
			return result;
		} else if (auto result = other->gt(this);
				   result.is_ok() && (result.unwrap() != not_implemented())) {
			return result;
		}
	} break;
	case RichCompare::Py_NE: {
		if (auto result = ne(other); result.is_ok() && (result.unwrap() != not_implemented())) {
			return result;
		} else if (auto result = other->ne(this);
				   result.is_ok() && (result.unwrap() != not_implemented())) {
			return result;
		}
	} break;
	}

	switch (op) {
	case RichCompare::Py_EQ: {
		return Ok(this == other ? py_true() : py_false());
	} break;
	case RichCompare::Py_NE: {
		return Ok(this != other ? py_true() : py_false());
	} break;
	default: {
		// op not supported
		return Err(type_error("'{}' not supported between instances of '{}' and '{}'",
			opstr[static_cast<size_t>(op)],
			type_prototype().__name__,
			other->type_prototype().__name__));
	}
	}
}

PyResult<PyObject *> PyObject::eq(const PyObject *other) const
{
	if (type_prototype().__eq__.has_value()) {
		return call_slot(*type_prototype().__eq__, this, other);
	}
	return Ok(not_implemented());
}

PyResult<PyObject *> PyObject::ge(const PyObject *other) const
{
	if (type_prototype().__eq__.has_value()) {
		return call_slot(*type_prototype().__ge__, this, other);
	}
	return Ok(not_implemented());
}

PyResult<PyObject *> PyObject::gt(const PyObject *other) const
{
	if (type_prototype().__gt__.has_value()) {
		return call_slot(*type_prototype().__gt__, this, other);
	}
	return Ok(not_implemented());
}

PyResult<PyObject *> PyObject::le(const PyObject *other) const
{
	if (type_prototype().__le__.has_value()) {
		return call_slot(*type_prototype().__le__, this, other);
	}
	return Ok(not_implemented());
}

PyResult<PyObject *> PyObject::lt(const PyObject *other) const
{
	if (type_prototype().__lt__.has_value()) {
		return call_slot(*type_prototype().__lt__, this, other);
	}
	return Ok(not_implemented());
}

PyResult<PyObject *> PyObject::ne(const PyObject *other) const
{
	if (type_prototype().__ne__.has_value()) {
		return call_slot(*type_prototype().__ne__, this, other);
	}
	return Ok(not_implemented());
}

PyResult<PyObject *> PyObject::getattribute(PyObject *attribute) const
{
	if (type_prototype().__getattribute__.has_value()) {
		return call_slot(*type_prototype().__getattribute__, this, attribute);
	}
	TODO();
}

PyResult<std::monostate> PyObject::setattribute(PyObject *attribute, PyObject *value)
{
	if (!as<PyString>(attribute)) {
		return Err(
			type_error("attribute name must be string, not '{}'", attribute->type()->to_string()));
	}

	if (auto setattr_method = type()->underlying_type().__setattribute__;
		setattr_method.has_value()) {
		return call_slot(*setattr_method, "", this, attribute, value);
	} else {
		return Err(type_error("'{}' object has no attributes (assign to {})",
			type()->underlying_type().__name__,
			attribute->to_string()));
	}
}

PyResult<PyObject *> PyObject::repr() const
{
	if (type_prototype().__repr__.has_value()) {
		return call_slot(*type_prototype().__repr__, this);
	}
	TODO();
}

PyResult<PyObject *> PyObject::str()
{
	if (type_prototype().__str__.has_value()) { return call_slot(*type_prototype().__str__, this); }
	return repr();
}

PyResult<int64_t> PyObject::hash() const
{
	if (type_prototype().__hash__.has_value()) {
		return call_slot(
			*type_prototype().__hash__, "__hash__ method should return an integer", this);
	} else {
		return __hash__();
	}
}

PyResult<PyObject *> PyObject::call(PyTuple *args, PyDict *kwargs)
{
	if (type_prototype().__call__.has_value()) {
		return call_slot(*type_prototype().__call__, this, args, kwargs);
	}
	return Err(type_error("'{}' object is not callable", type_prototype().__name__));
}


PyResult<PyObject *> PyObject::add(const PyObject *other) const
{
	if (type_prototype().__add__.has_value()) {
		return call_slot(*type_prototype().__add__, this, other);
	} else if (other->type_prototype().__add__.has_value()) {
		return call_slot(*other->type_prototype().__add__, other, this);
	} else if (auto seq = const_cast<PyObject *>(this)->as_sequence(); seq.is_ok()) {
		return seq.unwrap().concat(const_cast<PyObject *>(other));
	}
	return Err(type_error("unsupported operand type(s) for +: \'{}\' and \'{}\'",
		type_prototype().__name__,
		other->type_prototype().__name__));
}

PyResult<PyObject *> PyObject::subtract(const PyObject *other) const
{
	if (type_prototype().__sub__.has_value()) {
		return call_slot(*type_prototype().__sub__, this, other);
	}
	return Err(type_error("unsupported operand type(s) for -: \'{}\' and \'{}\'",
		type_prototype().__name__,
		other->type_prototype().__name__));
}

PyResult<PyObject *> PyObject::multiply(const PyObject *other) const
{
	if (type_prototype().__mul__.has_value()) {
		return call_slot(*type_prototype().__mul__, this, other);
	} else if (other->type_prototype().__mul__.has_value()) {
		return call_slot(*other->type_prototype().__mul__, other, this);
	}
	return Err(type_error("unsupported operand type(s) for *: \'{}\' and \'{}\'",
		type_prototype().__name__,
		other->type_prototype().__name__));
}

PyResult<PyObject *> PyObject::exp(const PyObject *other) const
{
	if (type_prototype().__exp__.has_value()) {
		return call_slot(*type_prototype().__exp__, this, other);
	}
	return Err(type_error("unsupported operand type(s) for **: \'{}\' and \'{}\'",
		type_prototype().__name__,
		other->type_prototype().__name__));
}

PyResult<PyObject *> PyObject::lshift(const PyObject *other) const
{
	if (type_prototype().__lshift__.has_value()) {
		return call_slot(*type_prototype().__lshift__, this, other);
	}
	return Err(type_error("unsupported operand type(s) for <<: \'{}\' and \'{}\'",
		type_prototype().__name__,
		other->type_prototype().__name__));
}

PyResult<PyObject *> PyObject::modulo(const PyObject *other) const
{
	if (type_prototype().__mod__.has_value()) {
		return call_slot(*type_prototype().__mod__, this, other);
	}
	return Err(type_error("unsupported operand type(s) for %: \'{}\' and \'{}\'",
		type_prototype().__name__,
		other->type_prototype().__name__));
}

PyResult<PyObject *> PyObject::and_(PyObject *other)
{
	if (type_prototype().__and__.has_value()) {
		return call_slot(*type_prototype().__and__, this, other);
	} else if (other->type_prototype().__and__.has_value()) {
		return call_slot(*other->type_prototype().__and__, other, this);
	}
	return Err(type_error("unsupported operand type(s) for &: \'{}\' and \'{}\'",
		type_prototype().__name__,
		other->type_prototype().__name__));
}

PyResult<PyObject *> PyObject::or_(PyObject *other)
{
	if (type_prototype().__or__.has_value()) {
		return call_slot(*type_prototype().__or__, this, other);
	} else if (other->type_prototype().__or__.has_value()) {
		return call_slot(*other->type_prototype().__or__, other, this);
	}
	return Err(type_error("unsupported operand type(s) for |: \'{}\' and \'{}\'",
		type_prototype().__name__,
		other->type_prototype().__name__));
}

PyResult<bool> PyObject::contains(PyObject *value)
{
	if (auto seq = as_sequence(); seq.is_ok()) {
		if (type_prototype().sequence_type_protocol->__contains__.has_value()) {
			return seq.unwrap().contains(value);
		}
	}
	auto it = iter();
	if (it.is_err()) { return Err(it.unwrap_err()); }

	auto n = it.unwrap()->next();

	while (n.is_ok()) {
		if (auto r = n.unwrap()->eq(value); r.is_ok()) {
			if (auto c = truthy(r.unwrap(), VirtualMachine::the().interpreter());
				c.is_ok() && c.unwrap()) {
				return Ok(true);
			} else if (c.is_err()) {
				return c;
			}
		} else if (r.is_err()) {
			return Err(r.unwrap_err());
		}
		n = it.unwrap()->next();
	}

	if (n.is_err() && (n.unwrap_err()->type() != stop_iteration()->type())) {
		return Err(n.unwrap_err());
	}
	return Ok(false);
}

PyResult<std::monostate> PyObject::delete_item(PyObject *key)
{
	if (auto mapping = as_mapping(); mapping.is_ok()) {
		if (type_prototype().mapping_type_protocol->__delitem__.has_value()) {
			return mapping.unwrap().delitem(key);
		}
	}

	// TODO: support sequence deletion

	return Err(type_error("'{}' object does not support item deletion", type_prototype().__name__));
}

PyResult<PyObject *> PyObject::abs() const
{
	if (type_prototype().__abs__.has_value()) { return call_slot(*type_prototype().__abs__, this); }
	return Err(type_error("bad operand type for abs(): '{}'", type_prototype().__name__));
}

PyResult<PyObject *> PyObject::neg() const
{
	if (type_prototype().__neg__.has_value()) { return call_slot(*type_prototype().__neg__, this); }
	return Err(type_error("bad operand type for unary -: '{}'", type_prototype().__name__));
}

PyResult<PyObject *> PyObject::pos() const
{
	if (type_prototype().__pos__.has_value()) { return call_slot(*type_prototype().__pos__, this); }
	return Err(type_error("bad operand type for unary +: '{}'", type_prototype().__name__));
}

PyResult<PyObject *> PyObject::invert() const
{
	if (type_prototype().__invert__.has_value()) {
		return call_slot(*type_prototype().__invert__, this);
	}
	return Err(type_error("bad operand type for unary ~: '{}'", type_prototype().__name__));
}


PyResult<bool> PyObject::bool_() const
{
	ASSERT(type_prototype().__bool__.has_value())
	return call_slot(*type_prototype().__bool__, "__bool__ should return bool", this);
}


PyResult<PyObject *> PyObject::iter() const
{
	if (type_prototype().__iter__.has_value()) {
		return call_slot(*type_prototype().__iter__, this);
	}

	return Err(type_error("'{}' object is not iterable", type()->name()));
}

PyResult<PyObject *> PyObject::next()
{
	if (type_prototype().__next__.has_value()) {
		return call_slot(*type_prototype().__next__, this);
	}

	return Err(type_error("'{}' object is not an iterator", type()->name()));
}

PyResult<PyObject *> PyObject::get(PyObject *instance, PyObject *owner) const
{
	if (type_prototype().__get__.has_value()) {
		return call_slot(*type_prototype().__get__, this, instance, owner);
	}
	TODO();
}

PyResult<PyObject *> PyObject::new_(PyTuple *args, PyDict *kwargs) const
{
	if (!as<PyType>(this)) {
		// FIXME: should be SystemError
		return Err(type_error("__new__() called with non-type 'self'"));
	}
	if (!args || args->size() < 1) {
		return Err(type_error("object.__new__(): not enough arguments"));
	}
	auto maybe_type_ = PyObject::from(args->elements()[0]);
	if (maybe_type_.is_err()) return maybe_type_;
	auto *maybe_type = maybe_type_.unwrap();
	if (!as<PyType>(maybe_type)) {
		return Err(type_error(
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
	args = args_.unwrap();

	if (type_prototype().__new__.has_value()) {
		return call_slot(*type_prototype().__new__, type, args, kwargs);
	}
	TODO();
}

PyResult<int32_t> PyObject::init(PyTuple *args, PyDict *kwargs)
{
	if (type_prototype().__init__.has_value()) {
		if (std::holds_alternative<InitSlotFunctionType>(*type_prototype().__init__)) {
			return std::get<InitSlotFunctionType>(*type_prototype().__init__)(this, args, kwargs);
		} else {
			std::vector<Value> new_args;
			new_args.reserve(args->size() + 1);
			new_args.push_back(this);
			new_args.insert(new_args.end(), args->elements().begin(), args->elements().end());
			auto args_ = PyTuple::create(new_args);
			if (args_.is_err()) return Err(args_.unwrap_err());
			auto result =
				std::get<PyObject *>(*type_prototype().__init__)->call(args_.unwrap(), kwargs);
			if (result.is_err()) return Err(result.unwrap_err());
			if (auto *obj = result.unwrap(); obj != py_none()) {
				return Err(type_error(
					"__init__() should return None, not '{}'", obj->type()->to_string()));
			}
			return Ok(0);
		}
	}
	TODO();
}

PyResult<PyObject *> PyObject::getitem(PyObject *key)
{
	if (as_mapping().is_ok() && type_prototype().mapping_type_protocol->__getitem__.has_value()) {
		return call_slot(*type_prototype().mapping_type_protocol->__getitem__, this, key);
	} else if (as_sequence().is_ok()) {
		// FIXME: add check to see if __getitem__ in sequence protocol is implemented
	}
	// TODO: could getitem be virtual and we override the logic below in PyType?
	if (as<PyType>(this)) {
		if (this == py::type()) { return PyGenericAlias::create(this, key); }
		auto method = get_method(PyString::create("__class_getitem__").unwrap());
		if (method.is_ok()) {
			return method.unwrap()->call(PyTuple::create(key).unwrap(), PyDict::create().unwrap());
		}
	}
	return Err(type_error("'{}' object is not subscriptable", type()->name()));
}


PyResult<PyObject *> PyObject::__eq__(const PyObject *other) const
{
	return Ok(this == other ? py_true() : py_false());
}

PyResult<PyObject *> PyObject::__getattribute__(PyObject *attribute) const
{
	if (!as<PyString>(attribute)) {
		return Err(
			type_error("attribute name must be a string, not {}", attribute->type()->to_string()));
	}

	auto *name = as<PyString>(attribute);

	auto descriptor_ = type()->lookup(name);
	if (descriptor_.is_err() && descriptor_.unwrap_err()->type() != AttributeError::static_type()) {
		return descriptor_;
	}

	bool descriptor_has_get = false;

	if (descriptor_.is_ok()) {
		auto *descriptor = descriptor_.unwrap();
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

	if (descriptor_has_get) {
		return descriptor_.unwrap()->get(const_cast<PyObject *>(this), type());
	}

	if (descriptor_.is_ok()) { return descriptor_; }

	return Err(attribute_error(
		"'{}' object has no attribute '{}'", type_prototype().__name__, name->to_string()));
}

PyResult<PyObject *> PyObject::get_attribute(PyObject *name) const
{
	if (!as<PyString>(name)) {
		return Err(
			type_error("attribute name must be a string, not {}", name->type()->to_string()));
	}
	const auto &getattribute_ = type()->underlying_type().__getattribute__;
	if (getattribute_.has_value()) { return getattribute(name); }

	return Err(
		attribute_error("'{}' object has no attribute '{}'", type()->name(), name->to_string()));
}

std::tuple<PyResult<PyObject *>, LookupAttrResult> PyObject::lookup_attribute(PyObject *name) const
{
	auto result = [&]() -> std::tuple<PyResult<PyObject *>, LookupAttrResult> {
		if (!as<PyString>(name)) {
			return { Err(type_error(
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
			return { Ok(py_none()), LookupAttrResult::NOT_FOUND };
		}
	}();

	if (std::get<0>(result).is_err()
		&& (std::get<0>(result).unwrap_err()->type() == AttributeError::static_type())) {
		return { Ok(py_none()), LookupAttrResult::NOT_FOUND };
	} else if (std::get<0>(result).is_err()) {
		return { std::get<0>(result), LookupAttrResult::NOT_FOUND };
	} else {
		return result;
	}
}


PyResult<PyObject *> PyObject::get_method(PyObject *name) const
{
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
	if (descriptor_.is_err() && descriptor_.unwrap_err()->type() != AttributeError::static_type()) {
		return descriptor_;
	}

	bool method_found = false;

	if (descriptor_.is_ok()) {
		auto *descriptor = descriptor_.unwrap();
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
		auto result = descriptor_.unwrap()->get(const_cast<PyObject *>(this), type());
		ASSERT(result.is_ok())
		return result;
	}

	return Err(attribute_error(
		"'{}' object has no attribute '{}'", type_prototype().__name__, name->to_string()));
}

PyResult<std::monostate> PyObject::__setattribute__(PyObject *attribute, PyObject *value)
{
	if (!as<PyString>(attribute)) {
		return Err(
			type_error("attribute name must be string, not '{}'", attribute->type()->to_string()));
	}

	if (auto descriptor_ = type()->lookup(attribute); descriptor_.is_ok()) {
		auto *descriptor = descriptor_.unwrap();
		const auto &descriptor_set = descriptor->type()->underlying_type().__set__;
		if (descriptor_set.has_value()) {
			return call_slot(*descriptor_set, "", descriptor, this, value);
		}
	}

	if (!m_attributes) {
		if (auto dict = PyDict::create(); dict.is_ok()) {
			m_attributes = dict.unwrap();
		} else {
			return Err(dict.unwrap_err());
		}
	}

	m_attributes->insert(attribute, value);

	return Ok(std::monostate{});
}

PyResult<bool> PyObject::__bool__() const { return Ok(true); }

PyResult<int64_t> PyObject::__hash__() const { return Ok(bit_cast<size_t>(this) >> 4); }

bool PyObject::is_callable() const { return type_prototype().__call__.has_value(); }

const std::string &PyObject::name() const { return type_prototype().__name__; }

PyResult<PyObject *> PyObject::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
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
				return Err(type_error(
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
				return Err(type_error("object() takes no arguments"));
			}
		}
	}
	// FIXME: if custom allocators are ever added, should call the type's allocator here
	auto *obj = VirtualMachine::the().heap().allocate<CustomPyObject>(type);
	if (!obj) { return Err(memory_error(sizeof(CustomPyObject))); }
	return Ok(obj);
}

PyResult<int32_t> PyObject::__init__(PyTuple *args, PyDict *kwargs)
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
				return Err(type_error(
					"object.__new__() takes exactly one argument (the type to instantiate)"));
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
				return Err(type_error("object() takes no arguments"));
			}
		}
	}
	return Ok(0);
}

std::string PyObject::to_string() const
{
	return fmt::format("PyObject at {}", static_cast<const void *>(this));
}

PyType *PyObject::type() const
{
	if (std::holds_alternative<PyType *>(m_type)) { return std::get<PyType *>(m_type); }
	ASSERT(false && "Static types should overload PyObject::type!");
}

namespace {

std::once_flag object_type_flag;

std::unique_ptr<TypePrototype> register_type()
{
	return std::move(klass<PyObject>("object")
						 .property(
							 "__class__",
							 [](PyObject *self) { return self->type(); },
							 [](PyObject *self, PyObject *value) -> PyResult<std::monostate> {
								 (void)self;
								 (void)value;
								 TODO();
							 })
						 .type);
}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyObject::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(object_type_flag, []() { type = ::register_type(); });
		return std::move(type);
	};
}