#include "PySuper.hpp"
#include "MemoryError.hpp"
#include "PyArgParser.hpp"
#include "PyCell.hpp"
#include "PyCode.hpp"
#include "PyFrame.hpp"
#include "PyList.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "RuntimeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

namespace py {

PySuper::PySuper() : PySuper(nullptr, nullptr, nullptr) {}

PySuper::PySuper(PyType *type, PyObject *object, PyType *object_type)
	: PyBaseObject(BuiltinTypes::the().super()), m_type(type), m_object(object),
	  m_object_type(object_type)
{}

PyResult<PyObject *> PySuper::__new__(const PyType *type, PyTuple *, PyDict *)
{
	ASSERT(type == super())
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate<PySuper>();
	if (!result) { return Err(memory_error(sizeof(PySuper))); }
	return Ok(result);
}

PyResult<int32_t> PySuper::__init__(PyTuple *args, PyDict *kwargs)
{
	auto parse_result = PyArgsParser<PyType *, PyObject *>::unpack_tuple(args,
		kwargs,
		"super",
		std::integral_constant<size_t, 0>{},
		std::integral_constant<size_t, 2>{},
		nullptr,
		nullptr);
	if (parse_result.is_err()) return Err(parse_result.unwrap_err());

	const auto [type_, obj] = parse_result.unwrap();

	if (!type_) {
		auto &interpreter = VirtualMachine::the().interpreter();
		auto *frame = interpreter.execution_frame();
		// This should never happen?
		if (!frame) { return Err(runtime_error("super(): no current frame")); }
		auto *code = frame->code();
		(void)code;
		TODO();
	}

	if (obj) {
		auto object_type_ = check(type_, obj);
		if (object_type_.is_err()) return Err(object_type_.unwrap_err());
		m_object_type = object_type_.unwrap();
	}

	m_type = type_;
	m_object = obj;

	return Ok(0);
}

PyResult<PyObject *> PySuper::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PySuper::__getattribute__(PyObject *name) const
{
	if (!m_object_type) { return PyObject::__getattribute__(name); }

	if (as<PyString>(name) && as<PyString>(name)->value() == "__class__") {
		return PyObject::__getattribute__(name);
	}

	ASSERT(m_object_type->__mro__);
	const auto &mro = m_object_type->__mro__->elements();

	auto it = std::find_if(mro.begin(), mro.end() - 1, [this](const auto &el) {
		ASSERT(std::holds_alternative<PyObject *>(el));
		return std::get<PyObject *>(el) == m_type;
	});
	if (it == mro.end() - 1) { return PyObject::__getattribute__(name); }

	const size_t type_index_in_mro = std::distance(mro.begin(), it) + 1;

	for (size_t index = type_index_in_mro; index < mro.size(); ++index) {
		auto el = PyObject::from(mro[index]);
		if (el.is_err()) return el;
		auto *candidate = el.unwrap();

		auto *attributes = candidate->attributes();
		ASSERT(attributes);

		if (auto it = attributes->map().find(name); it != attributes->map().end()) {
			auto res_ = PyObject::from(it->second);
			if (res_.is_err()) return res_;
			auto *res = res_.unwrap();

			if (res->type()->underlying_type().__get__) {
				return res->get(m_object == m_object_type ? nullptr : m_object, m_object_type);
			}

			return Ok(res);
		}
	}

	return PyObject::__getattribute__(name);
}

PyResult<PyType *> PySuper::check(PyType *type, PyObject *object)
{
	if (as<PyType>(object) && object->type()->issubclass(type)) { return Ok(as<PyType>(object)); }

	if (object->type()->issubclass(type)) {
		return Ok(object->type());
	} else {
		auto [obj_, lookup_result] =
			object->lookup_attribute(PyString::create("__class__").unwrap());
		if (obj_.is_err()) return Err(obj_.unwrap_err());

		auto *class_attribute = obj_.unwrap();
		if (as<PyType>(class_attribute) && class_attribute != object->type()) {
			if (as<PyType>(class_attribute)->issubclass(type)) {
				return Ok(as<PyType>(class_attribute));
			}
		}
	}

	return Err(type_error("super(type, obj: obj must be an instance or subtype of type"));
}

std::string PySuper::to_string() const
{
	if (m_object_type) {
		return fmt::format("<super: <class '{}'>, <{} object>>",
			m_type ? m_type->name() : "NULL",
			m_object_type->name());
	} else {
		return fmt::format("<super: <class '{}'>, NULL>", m_type ? m_type->name() : "NULL");
	}
}

void PySuper::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);

	if (m_type) visitor.visit(*m_type);
	if (m_object) visitor.visit(*m_object);
	if (m_object_type) visitor.visit(*m_object_type);
}

namespace {
	std::once_flag super_flag;
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PySuper::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(super_flag, []() {
			type = std::move(klass<PySuper>("super")
								 .attribute_readonly("__thisclass__", &PySuper::m_type)
								 .attribute_readonly("__self__", &PySuper::m_object)
								 .attribute_readonly("__self_class__", &PySuper::m_object_type)
								 .type);
		});
		return std::move(type);
	};
}

PyType *PySuper::type() const { return super(); }

}// namespace py