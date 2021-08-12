#include "AttributeError.hpp"
#include "PyNumber.hpp"
#include "PyObject.hpp"
#include "PyString.hpp"
#include "StopIterationException.hpp"

#include "bytecode/VM.hpp"
#include "bytecode/instructions/FunctionCall.hpp"
#include "interpreter/Interpreter.hpp"


size_t ValueHash::operator()(const Value &value) const
{
	return std::visit(
		overloaded{ [](const Number &number) -> size_t {
					   if (std::holds_alternative<double>(number.value)) {
						   return std::hash<double>{}(std::get<double>(number.value));
					   } else {
						   return std::hash<int64_t>{}(std::get<int64_t>(number.value));
					   }
				   },
			[](const String &s) -> size_t { return std::hash<std::string>{}(s.s); },
			[](const Bytes &b) -> size_t {
				return reinterpret_cast<size_t>(static_cast<const void *>(b.b.data()));
			},
			[](const Ellipsis &) -> size_t {
				return reinterpret_cast<size_t>(static_cast<const void *>(py_ellipsis().get()));
			},
			[](const NameConstant &c) -> size_t {
				if (std::holds_alternative<bool>(c.value)) {
					return std::get<bool>(c.value) ? 0 : 1;
				} else {
					return reinterpret_cast<size_t>(static_cast<const void *>(py_none().get()));
				}
			},
			[](const std::shared_ptr<PyObject> &obj) -> size_t {
				if (std::holds_alternative<std::monostate>(obj->slots().hash)) {
					return reinterpret_cast<size_t>(static_cast<const void *>(obj.get()));
				} else if (std::holds_alternative<HashSlotFunctionType>(obj->slots().hash)) {
					return std::get<HashSlotFunctionType>(obj->slots().hash)();
				} else {
					auto &vm = VirtualMachine::the();
					auto args = vm.heap().allocate<PyTuple>(std::vector{ obj });
					auto function_object = std::get<std::shared_ptr<PyFunction>>(obj->slots().hash);
					auto result =
						execute(vm, *vm.interpreter(), function_object, args, nullptr, nullptr);
					if (result->type() != PyObjectType::PY_NUMBER) {
						vm.interpreter()->raise_exception("");
						return 0;
					}
					return std::visit([](const auto &value) { return static_cast<size_t>(value); },
						as<PyNumber>(result)->value().value);
				}
			} },
		value);
}


bool ValueEqual::operator()(const Value &lhs_value, const Value &rhs_value) const
{
	return std::visit(
		overloaded{ [](const auto &lhs, const auto &rhs) { return lhs == rhs; },
			[](const std::shared_ptr<PyObject> &lhs, const std::shared_ptr<PyObject> &rhs) {
				if (lhs.get() == rhs.get()) { return true; }
				if (std::holds_alternative<std::monostate>(lhs->slots().richcompare)) {
					return lhs.get() == rhs.get();
				} else if (std::holds_alternative<RichCompareSlotFunctionType>(
							   lhs->slots().richcompare)) {
					return std::get<RichCompareSlotFunctionType>(lhs->slots().richcompare)(
							   rhs, RichCompare::Py_EQ)
						   == py_true();
				} else {
					TODO();
				}
			} },
		lhs_value,
		rhs_value);
}


template<> std::shared_ptr<PyObject> PyObject::from(const std::shared_ptr<PyObject> &value)
{
	return value;
}

template<> std::shared_ptr<PyObject> PyObject::from(const Number &value)
{
	return PyNumber::create(value);
}

template<> std::shared_ptr<PyObject> PyObject::from(const String &value)
{
	return PyString::create(value.s);
}

template<> std::shared_ptr<PyObject> PyObject::from(const Bytes &value)
{
	return PyBytes::create(value);
}

template<> std::shared_ptr<PyObject> PyObject::from(const Ellipsis &) { return py_ellipsis(); }

template<> std::shared_ptr<PyObject> PyObject::from(const NameConstant &value)
{
	if (auto none_value = std::get_if<NoneType>(&value.value)) {
		return py_none();
	} else {
		const bool bool_value = std::get<bool>(value.value);
		return bool_value ? py_true() : py_false();
	}
}

template<> std::shared_ptr<PyObject> PyObject::from(const Value &value)
{
	return std::visit([](const auto &v) { return PyObject::from(v); }, value);
}


PyObject::PyObject(PyObjectType type) : m_type(type)
{
	m_slots =
		Slots{ .repr = [this]() { return this->repr_impl(*VirtualMachine::the().interpreter()); },
			.iter = [this]() { return this->iter_impl(*VirtualMachine::the().interpreter()); },
			.hash = {},
			.richcompare = {} };
};


void PyObject::put(std::string name, std::shared_ptr<PyObject> value)
{
	m_attributes.insert_or_assign(name, value);
}


std::shared_ptr<PyObject> PyObject::get(std::string name, Interpreter &interpreter) const
{
	if (auto it = m_attributes.find(name); it != m_attributes.end()) { return it->second; }
	interpreter.raise_exception(attribute_error(
		fmt::format("'{}' object has no attribute '{}'", object_name(m_type), name)));
	return nullptr;
}


std::shared_ptr<PyObject> PyObject::repr_impl(Interpreter &) const
{
	return PyString::from(String{ fmt::format("<object at {}>", static_cast<const void *>(this)) });
}

std::shared_ptr<PyObject> PyObject::iter_impl(Interpreter &interpreter) const
{
	interpreter.raise_exception(
		fmt::format("TypeError: '{}' object is not iterable", object_name(type())));
	return nullptr;
}

std::shared_ptr<PyObject> PyObject::next_impl(Interpreter &interpreter)
{
	interpreter.raise_exception(
		fmt::format("TypeError: '{}' object is not an iterator", object_name(type())));
	return nullptr;
}

std::shared_ptr<PyObject> PyObject::len_impl(Interpreter &interpreter) const
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


std::shared_ptr<PyObject> PyNumber::repr_impl(Interpreter &) const
{
	return PyString::from(String{ to_string() });
}


std::string PyNameConstant::to_string() const
{
	if (std::holds_alternative<NoneType>(m_value.value)) {
		return "None";
	} else {
		bool bool_value = std::get<bool>(m_value.value);
		return bool_value ? "True" : "False";
	}
}

std::shared_ptr<PyObject> PyNameConstant::repr_impl(Interpreter &) const
{
	return PyString::from(String{ to_string() });
}


std::shared_ptr<PyObject> PyObject::add_impl(const std::shared_ptr<PyObject> &, Interpreter &) const
{
	return nullptr;
}


std::shared_ptr<PyObject> PyObject::subtract_impl(const std::shared_ptr<PyObject> &,
	Interpreter &) const
{
	return nullptr;
}


std::shared_ptr<PyObject> PyObject::multiply_impl(const std::shared_ptr<PyObject> &,
	Interpreter &) const
{
	return nullptr;
}


std::shared_ptr<PyObject> PyObject::exp_impl(const std::shared_ptr<PyObject> &, Interpreter &) const
{
	return nullptr;
}


std::shared_ptr<PyObject> PyObject::lshift_impl(const std::shared_ptr<PyObject> &,
	Interpreter &) const
{
	return nullptr;
}

std::shared_ptr<PyObject> PyObject::modulo_impl(const std::shared_ptr<PyObject> &,
	Interpreter &) const
{
	return nullptr;
}

std::shared_ptr<PyObject> PyObject::equal_impl(const std::shared_ptr<PyObject> &,
	Interpreter &) const
{
	return PyObject::from(NameConstant{ false });
}


std::shared_ptr<PyObject> PyObject::richcompare_impl(const std::shared_ptr<PyObject> &other,
	RichCompare op,
	Interpreter &interpreter) const
{
	static constexpr std::string_view opstrings[] = { "<", "<=", "==", "!=", ">", ">=" };
	switch (op) {
	case RichCompare::Py_EQ:
		return this == other.get() ? py_true() : py_false();
	case RichCompare::Py_NE:
		return this != other.get() ? py_true() : py_false();
	default:
		interpreter.raise_exception(
			"TypeError: '{}' not supported between instances of '{}' and '{}'",
			opstrings[static_cast<int>(op)],
			object_name(this->type()),
			object_name(other->type()));
	}
	return nullptr;
}


PyCode::PyCode(const size_t pos, const size_t register_count, std::vector<std::string> args)
	: PyObject(PyObjectType::PY_CODE), m_pos(pos), m_register_count(register_count),
	  m_args(std::move(args))
{
	// 	m_attributes["co_var"] = PyList::from(m_args);
}

PyFunction::PyFunction(std::string name, std::shared_ptr<PyCode> code)
	: PyObject(PyObjectType::PY_FUNCTION), m_name(std::move(name)), m_code(std::move(code))
{
	// m_attributes["__code__"] = m_code;
}


std::shared_ptr<PyObject> PyBytes::add_impl(const std::shared_ptr<PyObject> &obj,
	Interpreter &interpreter) const
{
	interpreter.raise_exception("TypeError: unsupported operand type(s) for +: \'{}\' and \'{}\'",
		object_name(type()),
		object_name(obj->type()));
	return nullptr;
}

std::shared_ptr<PyObject> PyEllipsis::add_impl(const std::shared_ptr<PyObject> &obj,
	Interpreter &interpreter) const
{
	interpreter.raise_exception("TypeError: unsupported operand type(s) for +: \'{}\' and \'{}\'",
		object_name(type()),
		object_name(obj->type()));
	return nullptr;
}


std::shared_ptr<PyObject> PyNameConstant::add_impl(const std::shared_ptr<PyObject> &obj,
	Interpreter &interpreter) const
{
	interpreter.raise_exception("TypeError: unsupported operand type(s) for +: \'{}\' and \'{}\'",
		object_name(type()),
		object_name(obj->type()));
	return nullptr;
}


std::string PyList::to_string() const
{
	std::ostringstream os;

	os << "[";
	auto it = m_elements.begin();
	while (std::next(it) != m_elements.end()) {
		std::visit([&os](const auto &value) { os << value << ", "; }, *it);
		std::advance(it, 1);
	}
	std::visit([&os](const auto &value) { os << value; }, *it);
	os << "]";

	return os.str();
}

std::shared_ptr<PyObject> PyList::repr_impl(Interpreter &) const
{
	return PyString::from(String{ to_string() });
}

std::shared_ptr<PyObject> PyList::iter_impl(Interpreter &) const
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyListIterator>(shared_from_this_as<PyList>());
}

std::string PyListIterator::to_string() const
{
	return fmt::format("<list_iterator at {}>", static_cast<const void *>(this));
}

std::shared_ptr<PyObject> PyListIterator::repr_impl(Interpreter &) const
{
	return PyString::from(String{ to_string() });
}

std::shared_ptr<PyObject> PyListIterator::next_impl(Interpreter &interpreter)
{
	if (m_current_index < m_pylist->elements().size())
		return std::visit([](const auto &element) { return PyObject::from(element); },
			m_pylist->elements()[m_current_index++]);
	interpreter.raise_exception(stop_iteration(""));
	return nullptr;
}


std::string PyTuple::to_string() const
{
	std::ostringstream os;

	os << "(";
	auto it = m_elements.begin();
	while (std::next(it) != m_elements.end()) {
		std::visit([&os](const auto &value) { os << value << ", "; }, *it);
		std::advance(it, 1);
	}
	std::visit([&os](const auto &value) { os << value; }, *it);
	os << ")";

	return os.str();
}

std::shared_ptr<PyObject> PyTuple::repr_impl(Interpreter &) const
{
	return PyString::from(String{ to_string() });
}

std::shared_ptr<PyObject> PyTuple::iter_impl(Interpreter &) const
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyTupleIterator>(shared_from_this_as<PyTuple>());
}


PyTupleIterator PyTuple::begin() const { return PyTupleIterator(shared_from_this_as<PyTuple>()); }


PyTupleIterator PyTuple::end() const
{
	return PyTupleIterator(shared_from_this_as<PyTuple>(), m_elements.size());
}

std::shared_ptr<PyObject> PyTuple::operator[](size_t idx) const
{
	return std::visit([](const auto &value) { return PyObject::from(value); }, m_elements[idx]);
}


std::string PyTupleIterator::to_string() const
{
	return fmt::format("<tuple_iterator at {}>", static_cast<const void *>(this));
}

std::shared_ptr<PyObject> PyTupleIterator::repr_impl(Interpreter &) const
{
	return PyString::from(String{ to_string() });
}

std::shared_ptr<PyObject> PyTupleIterator::next_impl(Interpreter &interpreter)
{
	if (m_current_index < m_pytuple->elements().size())
		return std::visit([](const auto &element) { return PyObject::from(element); },
			m_pytuple->elements()[m_current_index++]);
	interpreter.raise_exception(stop_iteration(""));
	return nullptr;
}

bool PyTupleIterator::operator==(const PyTupleIterator &other) const
{
	return m_pytuple.get() == other.m_pytuple.get() && m_current_index == other.m_current_index;
}

PyTupleIterator &PyTupleIterator::operator++()
{
	m_current_index++;
	return *this;
}

PyTupleIterator &PyTupleIterator::operator--()
{
	m_current_index--;
	return *this;
}

std::shared_ptr<PyObject> PyTupleIterator::operator*() const
{
	return std::visit([](const auto &element) { return PyObject::from(element); },
		m_pytuple->elements()[m_current_index]);
}

std::shared_ptr<PyString> PyString::create(const std::string &value)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyString>(value);
}

std::shared_ptr<PyNumber> PyNumber::create(const Number &number)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyNumber>(number);
}

std::shared_ptr<PyBytes> PyBytes::create(const Bytes &value)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyBytes>(value);
}

std::shared_ptr<PyEllipsis> PyEllipsis::create()
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate_static<PyEllipsis>();
}

std::shared_ptr<PyNameConstant> PyNameConstant::create(const NameConstant &value)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate_static<PyNameConstant>(value);
}


class Env
{
  public:
	static std::shared_ptr<PyObject> py_true() { return Env::instance().m_py_true; }
	static std::shared_ptr<PyObject> py_false() { return Env::instance().m_py_false; }
	static std::shared_ptr<PyObject> py_none()
	{
		spdlog::debug("none {}", (void *)Env::instance().m_py_none.get());
		return Env::instance().m_py_none;
	}

  private:
	std::shared_ptr<PyObject> m_py_true;
	std::shared_ptr<PyObject> m_py_false;
	std::shared_ptr<PyObject> m_py_none;

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

std::shared_ptr<PyObject> py_true() { return Env::py_true(); }
std::shared_ptr<PyObject> py_false() { return Env::py_false(); }
std::shared_ptr<PyObject> py_none() { return Env::py_none(); }


std::shared_ptr<PyObject> py_ellipsis()
{
	static std::shared_ptr<PyObject> ellipsis = nullptr;
	if (!ellipsis) { ellipsis = PyEllipsis::create(); }
	return ellipsis;
}