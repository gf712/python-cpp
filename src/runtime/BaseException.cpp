#include "BaseException.hpp"
#include "MemoryError.hpp"
#include "PyCode.hpp"
#include "PyFrame.hpp"
#include "PyTraceback.hpp"
#include "PyType.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "vm/VM.hpp"

namespace py {

template<> BaseException *as(PyObject *obj)
{
	ASSERT(types::base_exception());
	if (obj->type() == types::base_exception()) { return static_cast<BaseException *>(obj); }
	return nullptr;
}

template<> const BaseException *as(const PyObject *obj)
{
	ASSERT(types::base_exception());
	if (obj->type() == types::base_exception()) { return static_cast<const BaseException *>(obj); }
	return nullptr;
}

BaseException::BaseException(PyType *type) : PyBaseObject(type->underlying_type()) {}

BaseException::BaseException(PyType *type, PyTuple *args) : PyBaseObject(type), m_args(args) {}

BaseException::BaseException(PyTuple *args)
	: PyBaseObject(types::BuiltinTypes::the().base_exception()), m_args(args)
{}

BaseException::BaseException(const TypePrototype &type, PyTuple *args)
	: PyBaseObject(type), m_args(args)
{}

PyResult<BaseException *> BaseException::create(PyTuple *args)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate<BaseException>(args);
	if (!result) { return Err(memory_error(sizeof(BaseException))); }
	return Ok(result);
}

PyResult<PyObject *> BaseException::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	ASSERT(type == types::base_exception());
	ASSERT(!kwargs || kwargs->map().empty())
	if (auto result = BaseException::create(args); result.is_ok()) {
		return Ok(static_cast<PyObject *>(result.unwrap()));
	} else {
		return Err(result.unwrap_err());
	}
}

PyResult<int32_t> BaseException::__init__(PyTuple *args, PyDict *kwargs)
{
	ASSERT(!kwargs || kwargs->map().empty())// takes no keyword arguments
	m_args = args;
	return Ok(0);
}

PyType *BaseException::static_type() const
{
	ASSERT(types::base_exception());
	return types::base_exception();
}

PyType *BaseException::class_type()
{
	ASSERT(types::base_exception())
	return types::base_exception();
}

std::string BaseException::what() const { return BaseException::to_string(); }

void BaseException::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_args) visitor.visit(*m_args);
	if (m_traceback) visitor.visit(*m_traceback);
}

std::string BaseException::to_string() const
{
	if (m_args) {
		if (m_args->size() == 1) {
			auto obj_ = PyObject::from(m_args->elements()[0]);
			ASSERT(obj_.is_ok())
			return obj_.unwrap()->to_string();
		} else {
			return m_args->to_string();
		}
	} else {
		return "";
	}
}

std::string BaseException::format_traceback() const
{
	std::ostringstream out;
	out << "Traceback (most recent call last):\n";
	auto *tb = m_traceback;
	while (tb) {
		out << fmt::format("  File \"{}\", line {}, in {}\n",
			"TODO",
			tb->m_tb_lineno,
			tb->m_tb_frame->code()->name());
		out << "    TODO -> print line\n";
		tb = tb->m_tb_next;
	}
	out << type()->name() << ": " << what() << "\n";
	return out.str();
}

PyResult<PyObject *> BaseException::__repr__() const
{
	if (auto result = PyString::create(fmt::format("{}({})", type()->name(), what()));
		result.is_ok()) {
		return Ok(static_cast<PyObject *>(result.unwrap()));
	} else {
		return Err(result.unwrap_err());
	}
}

namespace {

	std::once_flag base_exception_flag;

	std::unique_ptr<TypePrototype> register_base_exception()
	{
		return std::move(klass<BaseException>("BaseException").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> BaseException::type_factory()
{
	return []() {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(base_exception_flag, []() { type = register_base_exception(); });
		return std::move(type);
	};
}
}// namespace py
