#include "Interpreter.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/StopIterationException.hpp"
#include "runtime/Value.hpp"
#include "runtime/PyRange.hpp"

#include <iostream>

Interpreter::Interpreter() : m_current_frame(ExecutionFrame::create(nullptr)) {}

namespace {
std::optional<int64_t> to_integer(const std::shared_ptr<PyObject> &obj, Interpreter &interpreter)
{
	if (auto pynumber = as<PyObjectNumber>(obj)) {
		if (auto int_value = std::get_if<int64_t>(&pynumber->value().value)) { return *int_value; }
		interpreter.raise_exception(
			"TypeError: '{}' object cannot be interpreted as an integer", object_name(obj->type()));
	}
	return {};
}
}// namespace

void Interpreter::setup()
{
	allocate_object<PyNativeFunction>(
		"print", [this](const std::shared_ptr<PyTuple> &args) {
			const std::string separator = " ";
			for (const auto &arg : *args) {
				std::cout << arg->repr_impl(*this)->to_string() << separator;
			}
			// make sure this is flushed immediately
			std::cout << std::endl;
			return py_none();
		});

	allocate_object<PyNativeFunction>("iter", [this](const std::shared_ptr<PyTuple> &args) {
		const auto &arg = args->operator[](0);
		return arg->iter_impl(*this);
	});

	allocate_object<PyNativeFunction>("next", [this](const std::shared_ptr<PyTuple> &args) {
		const auto &arg = args->operator[](0);
		auto result = arg->next_impl(*this);
		if (!result) { raise_exception(stop_iteration("")); }
		return result;
	});

	allocate_object<PyNativeFunction>("range", [this](const std::shared_ptr<PyTuple> &args) {
		const auto &arg = args->operator[](0);
		auto &heap = VirtualMachine::the().heap();
		if (auto pynumber = to_integer(arg, *this)) {
			return std::static_pointer_cast<PyObject>(heap.allocate<PyRange>(*pynumber));
		}
		return py_none();
	});
}

void Interpreter::unwind()
{
	auto raised_exception = m_current_frame->exception();
	while (!m_current_frame->catch_exception(raised_exception)) {
		// don't unwind beyond the main frame
		if (!m_current_frame->parent()) {
			// uncaught exception
			std::cout
				<< std::static_pointer_cast<BaseException>(m_current_frame->exception())->what()
				<< '\n';
			break;
		}
		m_current_frame = m_current_frame->parent();
	}
	m_current_frame->set_exception(nullptr);
}