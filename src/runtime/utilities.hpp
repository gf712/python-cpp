#pragma once

#include "PyFrame.hpp"
#include "PyObject.hpp"
#include "StopIteration.hpp"
#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"

namespace py {

namespace detail {
	template<typename T>
	concept has_output_iterator_error = requires(const T iterator) {
		{ iterator.last_error() } -> std::convertible_to<BaseException *>;
	};
}// namespace detail

template<typename OutputIterator>
PyResult<std::monostate> from_iterable(PyObject *iterable, OutputIterator result)
{
	static constexpr bool OutputIteratorCanError =
		detail::has_output_iterator_error<OutputIterator>;

	auto &vm = VirtualMachine::the();
	auto exc = vm.interpreter().execution_frame()->exception_info();

	auto iterator = iterable->iter();
	if (iterator.is_err()) return Err(iterator.unwrap_err());

	auto value = iterator.unwrap()->next();

	while (value.is_ok()) {
		result = value.unwrap();
		if constexpr (OutputIteratorCanError) {
			if (result.last_error()) { return Err(result.last_error()); }
		}
		value = iterator.unwrap()->next();
	}

	if (value.unwrap_err()->type() != stop_iteration()->type()) { return Err(value.unwrap_err()); }

	auto current_exc = vm.interpreter().execution_frame()->exception_info();
	if (current_exc.has_value()) {
		auto *popped_exc = vm.interpreter().execution_frame()->pop_exception();
		if (exc.has_value()) {
			ASSERT(popped_exc == exc->exception);
		} else {
			ASSERT(!vm.interpreter().execution_frame()->exception_info().has_value());
		}
	}

	return Ok(std::monostate{});
}

template<typename FnReturnType, typename... FnArgs>
bool compare_slot_address(
	const std::optional<std::variant<std::function<FnReturnType(FnArgs...)>, PyObject *>> &lhs,
	const std::optional<std::variant<std::function<FnReturnType(FnArgs...)>, PyObject *>> &rhs)
{
	using FnType = std::function<FnReturnType(FnArgs...)>;
	using FnPointerType = FnReturnType (*)(FnArgs...);
	if (!lhs.has_value() || !rhs.has_value()) { return false; }
	if (lhs->index() != rhs->index()) { return false; }
	if (std::holds_alternative<FnType>(*lhs)) {
		ASSERT(std::get<FnType>(*lhs).template target<FnPointerType>());
		ASSERT(std::get<FnType>(*rhs).template target<FnPointerType>());
		return std::get<FnType>(*lhs).template target<FnPointerType>()
			   == std::get<FnType>(*rhs).template target<FnPointerType>();
	}
	return std::get<PyObject *>(*lhs) == std::get<PyObject *>(*rhs);
}

}// namespace py
