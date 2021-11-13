#include "PyObject.hpp"

#include "AttributeError.hpp"
#include "PyFunction.hpp"
#include "PyNumber.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "StopIterationException.hpp"

#include "executable/bytecode/instructions/FunctionCall.hpp"
#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"


size_t ValueHash::operator()(const Value &value) const
{
	const auto result = std::visit(
		overloaded{ [](const Number &number) -> size_t {
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
				if (std::holds_alternative<std::monostate>(obj->slots().hash)) {
					return bit_cast<size_t>(obj);
				} else if (std::holds_alternative<HashSlotFunctionType>(obj->slots().hash)) {
					return std::get<HashSlotFunctionType>(obj->slots().hash)();
				} else {
					auto &vm = VirtualMachine::the();
					auto *args = PyTuple::create(std::vector{ obj });
					auto *function_object = std::get<PyFunction *>(obj->slots().hash);
					auto *result =
						execute(vm.interpreter(), function_object, args, nullptr, nullptr);
					if (result->type() != PyObjectType::PY_NUMBER) {
						vm.interpreter().raise_exception("");
						return 0;
					}
					return std::visit([](const auto &value) { return static_cast<size_t>(value); },
						as<PyNumber>(result)->value().value);
				}
			} },
		value);
	return result;
}


bool ValueEqual::operator()(const Value &lhs_value, const Value &rhs_value) const
{
	const auto result = std::visit(
		overloaded{ [](PyObject *const lhs, PyObject *const rhs) {
					   if (lhs == rhs) { return true; }
					   if (std::holds_alternative<std::monostate>(lhs->slots().richcompare)) {
						   return lhs == rhs;
					   } else if (std::holds_alternative<RichCompareSlotFunctionType>(
									  lhs->slots().richcompare)) {
						   auto result = std::get<RichCompareSlotFunctionType>(
							   lhs->slots().richcompare)(rhs, RichCompare::Py_EQ);
						   return result == py_true();
					   } else {
						   TODO();
					   }
				   },
			[](PyObject *const lhs, const auto &rhs) { return lhs == rhs; },
			[](const auto &lhs, PyObject *const rhs) { return lhs == rhs; },
			[](const auto &lhs, const auto &rhs) { return lhs == rhs; } },
		lhs_value,
		rhs_value);
	return result;
}


template<> PyObject *PyObject::from(PyObject *const &value) { return value; }

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


PyObject::PyObject(PyObjectType type) : Cell(), m_type(type)
{
	m_slots =
		Slots{ .repr = [this]() { return this->repr_impl(VirtualMachine::the().interpreter()); },
			.iter = [this]() { return this->iter_impl(VirtualMachine::the().interpreter()); },
			.hash = {},
			.richcompare = {} };
};

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
	interpreter.raise_exception(attribute_error(
		fmt::format("'{}' object has no attribute '{}'", object_name(m_type), name)));
	return nullptr;
}


PyObject *PyObject::repr_impl(Interpreter &) const
{
	return PyString::from(String{ fmt::format("<object at {}>", static_cast<const void *>(this)) });
}

PyObject *PyObject::iter_impl(Interpreter &interpreter) const
{
	interpreter.raise_exception(
		fmt::format("TypeError: '{}' object is not iterable", object_name(type())));
	return nullptr;
}

PyObject *PyObject::next_impl(Interpreter &interpreter)
{
	interpreter.raise_exception(
		fmt::format("TypeError: '{}' object is not an iterator", object_name(type())));
	return nullptr;
}

PyObject *PyObject::len_impl(Interpreter &interpreter) const
{
	interpreter.raise_exception(
		fmt::format("TypeError: object of type '{}' has no len()", object_name(type())));
	return nullptr;
}


size_t PyObject::hash_impl(Interpreter &interpreter) const
{
	interpreter.raise_exception(
		fmt::format("TypeError: object of type '{}' has no hash()", object_name(type())));
	return 0;
}


PyObject *PyNumber::repr_impl(Interpreter &) const { return PyString::from(String{ to_string() }); }


std::string PyNameConstant::to_string() const
{
	if (std::holds_alternative<NoneType>(m_value.value)) {
		return "None";
	} else {
		bool bool_value = std::get<bool>(m_value.value);
		return bool_value ? "True" : "False";
	}
}

PyObject *PyNameConstant::repr_impl(Interpreter &) const
{
	return PyString::from(String{ to_string() });
}

PyObject *PyNameConstant::truthy(Interpreter &) const
{
	if (std::holds_alternative<NoneType>(m_value.value)) {
		return py_false();
	} else {
		return std::get<bool>(m_value.value) ? py_true() : py_false();
	}
}

PyObject *PyObject::add_impl(const PyObject *, Interpreter &) const { return nullptr; }


PyObject *PyObject::subtract_impl(const PyObject *, Interpreter &) const { return nullptr; }


PyObject *PyObject::multiply_impl(const PyObject *, Interpreter &) const { return nullptr; }


PyObject *PyObject::exp_impl(const PyObject *, Interpreter &) const { return nullptr; }


PyObject *PyObject::lshift_impl(const PyObject *, Interpreter &) const { return nullptr; }

PyObject *PyObject::modulo_impl(const PyObject *, Interpreter &) const { return nullptr; }

PyObject *PyObject::equal_impl(const PyObject *other, Interpreter &) const
{
	return this == other ? py_true() : py_false();
}


PyObject *PyObject::less_than_impl(const PyObject *other, Interpreter &interpreter) const
{
	interpreter.raise_exception("TypeError: '<' not supported between instances of '{}' and '{}'",
		object_name(this->type()),
		object_name(other->type()));
	return nullptr;
}

PyObject *PyObject::less_than_equal_impl(const PyObject *other, Interpreter &interpreter) const
{
	interpreter.raise_exception("TypeError: '<=' not supported between instances of '{}' and '{}'",
		object_name(this->type()),
		object_name(other->type()));
	return nullptr;
}

PyObject *PyObject::greater_than_impl(const PyObject *other, Interpreter &interpreter) const
{
	interpreter.raise_exception("TypeError: '>' not supported between instances of '{}' and '{}'",
		object_name(this->type()),
		object_name(other->type()));
	return nullptr;
}

PyObject *PyObject::greater_than_equal_impl(const PyObject *other, Interpreter &interpreter) const
{
	interpreter.raise_exception("TypeError: '>=' not supported between instances of '{}' and '{}'",
		object_name(this->type()),
		object_name(other->type()));
	return nullptr;
}

PyObject *PyObject::richcompare_impl(const PyObject *other,
	RichCompare op,
	Interpreter &interpreter) const
{
	switch (op) {
	case RichCompare::Py_EQ:
		return equal_impl(other, interpreter);
	case RichCompare::Py_NE:
		return equal_impl(other, interpreter) ? py_false() : py_true();
	case RichCompare::Py_LE:
		return less_than_equal_impl(other, interpreter);
	case RichCompare::Py_LT:
		return less_than_impl(other, interpreter);
	case RichCompare::Py_GE:
		return greater_than_equal_impl(other, interpreter);
	case RichCompare::Py_GT:
		return greater_than_impl(other, interpreter);
	}
	ASSERT_NOT_REACHED()
}

PyObject *PyObject::truthy(Interpreter &) const { return py_true(); }

PyObject *PyBytes::add_impl(const PyObject *obj, Interpreter &interpreter) const
{
	interpreter.raise_exception("TypeError: unsupported operand type(s) for +: \'{}\' and \'{}\'",
		object_name(type()),
		object_name(obj->type()));
	return nullptr;
}

PyObject *PyEllipsis::add_impl(const PyObject *obj, Interpreter &interpreter) const
{
	interpreter.raise_exception("TypeError: unsupported operand type(s) for +: \'{}\' and \'{}\'",
		object_name(type()),
		object_name(obj->type()));
	return nullptr;
}


PyObject *PyNameConstant::add_impl(const PyObject *obj, Interpreter &interpreter) const
{
	interpreter.raise_exception("TypeError: unsupported operand type(s) for +: \'{}\' and \'{}\'",
		object_name(type()),
		object_name(obj->type()));
	return nullptr;
}


PyString *PyString::create(const std::string &value)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyString>(value);
}

PyNumber *PyNumber::create(const Number &number)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyNumber>(number);
}

PyBytes *PyBytes::create(const Bytes &value)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyBytes>(value);
}

PyEllipsis *PyEllipsis::create()
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate_static<PyEllipsis>().get();
}

PyNameConstant *PyNameConstant::create(const NameConstant &value)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate_static<PyNameConstant>(value).get();
}


class Env
{
  public:
	static PyObject *py_true() { return Env::instance().m_py_true; }
	static PyObject *py_false() { return Env::instance().m_py_false; }
	static PyObject *py_none() { return Env::instance().m_py_none; }

  private:
	PyObject *m_py_true;
	PyObject *m_py_false;
	PyObject *m_py_none;

	static Env &instance()
	{
		static Env instance{};
		return instance;
	}

	Env()
	{
		m_py_true = PyNameConstant::create(NameConstant{ true });
		m_py_false = PyNameConstant::create(NameConstant{ false });
		m_py_none = PyNameConstant::create(NameConstant{ NoneType{} });
	}
};

PyObject *py_true() { return Env::py_true(); }
PyObject *py_false() { return Env::py_false(); }
PyObject *py_none() { return Env::py_none(); }


PyObject *py_ellipsis()
{
	static PyObject *ellipsis = nullptr;
	if (!ellipsis) { ellipsis = PyEllipsis::create(); }
	return ellipsis;
}