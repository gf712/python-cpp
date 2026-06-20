#include "BaseException.hpp"
#include "MemoryError.hpp"
#include "PyCode.hpp"
#include "PyFrame.hpp"
#include "PyNone.hpp"
#include "PyString.hpp"
#include "PyTraceback.hpp"
#include "PyTuple.hpp"
#include "PyType.hpp"
#include "SourceManager.hpp"
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
	ASSERT(!kwargs || kwargs->map().empty());
	if (auto result = BaseException::create(args); result.is_ok()) {
		return Ok(static_cast<PyObject *>(result.unwrap()));
	} else {
		return Err(result.unwrap_err());
	}
}

PyResult<int32_t> BaseException::__init__(PyTuple *args, PyDict *kwargs)
{
	// takes no keyword arguments
	ASSERT(!kwargs || kwargs->map().empty());
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
	ASSERT(types::base_exception());
	return types::base_exception();
}

std::string BaseException::what() const { return BaseException::to_string(); }

void BaseException::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	if (m_args) visitor.visit(*m_args);
	if (m_dict) visitor.visit(*m_dict);
	if (m_traceback) visitor.visit(*m_traceback);
	if (m_context) visitor.visit(*m_context);
	if (m_cause) visitor.visit(*m_cause);
}

std::string BaseException::to_string() const
{
	if (m_args) {
		if (m_args->size() == 1) {
			auto obj_ = PyObject::from(m_args->elements()[0]);
			ASSERT(obj_.is_ok());
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
		const auto &filename = tb->m_tb_frame->code()->m_filename;
		out << fmt::format("  File \"{}\", line {}, in {}\n",
			filename,
			tb->m_tb_lineno,
			tb->m_tb_frame->code()->name());
		const auto source = SourceManager::the().line(filename, tb->m_tb_lineno);
		const auto trimmed = SourceManager::strip_leading_whitespace(source);
		if (!trimmed.empty()) { out << "    " << trimmed << "\n"; }
		tb = tb->m_tb_next;
	}
	out << type()->name() << ": " << what() << "\n";
	return out.str();
}

PyResult<PyObject *> BaseException::__repr__() const
{
	std::string args_part;
	if (m_args && m_args->size() == 1) {
		auto r = repr_value(m_args->elements()[0]);
		if (r.is_err()) { return r; }
		args_part = fmt::format("({})", r.unwrap()->to_string());
	} else if (m_args) {
		auto r = m_args->repr();
		if (r.is_err()) { return r; }
		args_part = r.unwrap()->to_string();
	} else {
		args_part = "()";
	}
	return PyString::create(fmt::format("{}{}", type()->name(), args_part));
}

PyResult<PyObject *> BaseException::__str__() const { return PyString::create(to_string()); }

namespace {

	std::once_flag base_exception_flag;

	std::unique_ptr<TypePrototype> register_base_exception()
	{
		return std::move(klass<BaseException>("BaseException")
				.property_readonly("args",
					[](BaseException *self) -> PyResult<PyObject *> {
						// args is always a tuple (empty when constructed without args).
						if (auto args = self->args()) { return Ok(args); }
						return PyTuple::create();
					})
				.property_readonly("__traceback__",
					[](BaseException *self) -> PyResult<PyObject *> {
						return Ok(self->traceback() ? self->traceback() : py_none());
					})
				.type);
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
