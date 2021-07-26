#include "Interpreter.hpp"

#include "bytecode/instructions/FunctionCall.hpp"

#include "runtime/PyObject.hpp"
#include "runtime/StopIterationException.hpp"
#include "runtime/Value.hpp"
#include "runtime/PyRange.hpp"
#include "runtime/CustomPyObject.hpp"

#include <iostream>

Interpreter::Interpreter() {}

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

std::shared_ptr<PyFunction> make_function(const std::string &function_name,
	int64_t function_id,
	const std::vector<std::string> &argnames)
{
	auto &vm = VirtualMachine::the();
	auto code = vm.heap().allocate<PyCode>(
		vm.function_offset(function_id), vm.function_register_count(function_id), argnames);
	return vm.heap().allocate<PyFunction>(function_name, std::static_pointer_cast<PyCode>(code));
}
}// namespace

void Interpreter::setup()
{
	m_current_frame = ExecutionFrame::create(nullptr, "__main__");

	allocate_object<PyNativeFunction>(
		"print", "print", [this](const std::shared_ptr<PyTuple> &args) {
			const std::string separator = " ";
			for (const auto &arg : *args) {
				std::cout << arg->repr_impl(*this)->to_string() << separator;
			}
			// make sure this is flushed immediately
			std::cout << std::endl;
			return py_none();
		});

	allocate_object<PyNativeFunction>("iter", "iter", [this](const std::shared_ptr<PyTuple> &args) {
		const auto &arg = args->operator[](0);
		return arg->iter_impl(*this);
	});

	allocate_object<PyNativeFunction>("next", "next", [this](const std::shared_ptr<PyTuple> &args) {
		const auto &arg = args->operator[](0);
		auto result = arg->next_impl(*this);
		if (!result) { raise_exception(stop_iteration("")); }
		return result;
	});

	allocate_object<PyNativeFunction>(
		"range", "range", [this](const std::shared_ptr<PyTuple> &args) {
			const auto &arg = args->operator[](0);
			auto &heap = VirtualMachine::the().heap();
			if (auto pynumber = to_integer(arg, *this)) {
				return std::static_pointer_cast<PyObject>(heap.allocate<PyRange>(*pynumber));
			}
			return py_none();
		});

	allocate_object<PyNativeFunction>(
		"__build_class__", "__build_class__", [this](const std::shared_ptr<PyTuple> &args) {
			const auto &class_name = args->operator[](0);
			const auto &function_location = args->operator[](1);
			std::cout << fmt::format(
				"__build_class__({}, {})", class_name->to_string(), function_location->to_string())
					  << '\n';

			ASSERT(as<PyString>(class_name))
			auto class_name_as_string = as<PyString>(class_name)->value();

			ASSERT(as<PyObjectNumber>(function_location))
			auto pynumber = as<PyObjectNumber>(function_location)->value();
			ASSERT(std::get_if<int64_t>(&pynumber.value))
			auto function_id = std::get<int64_t>(pynumber.value);

			auto pyfunc =
				make_function(class_name_as_string, function_id, std::vector<std::string>{});

			auto &vm = VirtualMachine::the();

			return vm.heap().allocate<PyNativeFunction>(class_name_as_string,
				[class_name, this, pyfunc, class_name_as_string](
					const std::shared_ptr<PyTuple> &args) {
					std::vector args_vector{ class_name };
					for (const auto &arg : *args) { args_vector.push_back(arg); }

					auto &vm = VirtualMachine::the();
					auto class_args = vm.heap().allocate<PyTuple>(args_vector);

					execute(vm, *this, pyfunc, class_args);

					CustomPyObjectContext ctx{ class_name_as_string };

					return vm.heap().allocate<CustomPyObject>(ctx, std::shared_ptr<PyTuple>{});
				});
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