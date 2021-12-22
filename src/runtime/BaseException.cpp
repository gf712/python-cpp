#include "BaseException.hpp"
#include "PyType.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"


PyType *BaseException::s_base_exception_type = nullptr;

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

std::string BaseException::what() const { return to_string(); }

void BaseException::visit_graph(Visitor &visitor)
{
	if (m_args) visitor.visit(*m_args);
}

std::string BaseException::to_string() const
{
	// const auto &exception_name = m_type_prototype.__name__;
	// if (m_args) {
	// 	const auto args = m_args->to_string();
	// 	std::string_view msg{ args.begin() + 1, args.end() - 1 };
	// 	return fmt::format("{}({})", exception_name, msg);
	// } else {
	// 	return fmt::format("{}()", exception_name);
	// }
	ASSERT(m_args->size() == 1)
	const auto &error_msg = PyObject::from(m_args->elements()[0])->to_string();

	const auto &exception_name = m_type_prototype.__name__;
	return fmt::format("{}: {}", exception_name, error_msg);
}

PyType *BaseException::register_type(PyModule *module)
{
	if (!s_base_exception_type) {
		s_base_exception_type = klass<BaseException>(module, "BaseException").finalize();
	}
	return s_base_exception_type;
}