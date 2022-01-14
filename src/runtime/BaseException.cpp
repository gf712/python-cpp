#include "BaseException.hpp"
#include "PyType.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

using namespace py;

PyType *BaseException::s_base_exception_type = nullptr;

template<> BaseException *py::as(PyObject *obj)
{
	ASSERT(BaseException::s_base_exception_type)
	if (obj->type() == BaseException::s_base_exception_type) {
		return static_cast<BaseException *>(obj);
	}
	return nullptr;
}

template<> const BaseException *py::as(const PyObject *obj)
{
	ASSERT(BaseException::s_base_exception_type)
	if (obj->type() == BaseException::s_base_exception_type) {
		return static_cast<const BaseException *>(obj);
	}
	return nullptr;
}

BaseException::BaseException(PyTuple *args)
	: PyBaseObject(s_base_exception_type->underlying_type()), m_args(args)
{}

BaseException::BaseException(const TypePrototype &type, PyTuple *args)
	: PyBaseObject(type), m_args(args)
{}

PyType *BaseException::type() const
{
	ASSERT(s_base_exception_type)
	return s_base_exception_type;
}

std::string BaseException::what() const { return BaseException::to_string(); }

void BaseException::visit_graph(Visitor &visitor)
{
	if (m_args) visitor.visit(*m_args);
}

std::string BaseException::to_string() const
{
	if (m_args->size() == 1) {
		return PyObject::from(m_args->elements()[0])->to_string();
	} else {
		return m_args->to_string();
	}
}

PyObject *BaseException::__repr__() const
{
	return PyString::create(fmt::format("{}({})", m_type_prototype.__name__, what()));
}

PyType *BaseException::register_type(PyModule *module)
{
	if (!s_base_exception_type) {
		s_base_exception_type = klass<BaseException>(module, "BaseException").finalize();
	}
	return s_base_exception_type;
}