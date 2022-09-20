#pragma once

#include "PyDict.hpp"
#include "PyNone.hpp"
#include "PyTuple.hpp"
#include "TypeError.hpp"

namespace py {

template<typename... ArgTypes> struct PyArgsParser
{
  private:
	template<size_t Idx,
		size_t MinSize,
		size_t MaxSize,
		typename ResultType,
		typename... DefaultArgs>
	static constexpr PyResult<std::monostate> unpack_tuple_helper(const std::vector<Value> &args,
		std::string_view function_name,
		std::integral_constant<size_t, MinSize> min_size,
		std::integral_constant<size_t, MaxSize> max_size,
		ResultType &result,
		DefaultArgs &&...default_args)
	{
		using ExpectedType = std::tuple_element_t<Idx, ResultType>;
		if constexpr (std::is_base_of_v<PyObject,
						  std::remove_pointer_t<std::remove_cv_t<ExpectedType>>>) {
			if (args.size() > Idx) {
				using PyObjectType = std::remove_pointer_t<std::remove_cv_t<ExpectedType>>;
				const auto &arg = PyObject::from(args[Idx]);
				if (arg.is_err()) return Err(arg.unwrap_err());
				if constexpr (std::is_same_v<PyObject, PyObjectType>) {
					std::get<Idx>(result) = arg.unwrap();
				} else {
					if (!as<PyObjectType>(arg.unwrap())) return Err(type_error("Unexpected type"));
					std::get<Idx>(result) = as<PyObjectType>(arg.unwrap());
				}
			} else {
				if constexpr (Idx >= MinSize && (Idx - MinSize) < sizeof...(DefaultArgs)) {
					std::get<Idx>(result) = std::get<Idx - MinSize>(
						std::forward_as_tuple(std::forward<DefaultArgs>(default_args)...));
				} else {
					TODO();
				}
			}

			if constexpr (Idx + 1 == std::tuple_size_v<ResultType>) {
				return Ok(std::monostate{});
			} else {
				return unpack_tuple_helper<Idx + 1>(args,
					function_name,
					min_size,
					max_size,
					result,
					std::forward<DefaultArgs>(default_args)...);
			}
		} else {
			TODO();
		}
	}

  public:
	template<size_t MinSize, size_t MaxSize, typename... DefaultArgs>
	static constexpr PyResult<std::tuple<ArgTypes...>> unpack_tuple(PyTuple *args,
		PyDict *kwargs,
		std::string_view function_name,
		std::integral_constant<size_t, MinSize> min_size,
		std::integral_constant<size_t, MaxSize> max_size,
		DefaultArgs &&...default_values)
	{
		if constexpr (max_size() - min_size() > sizeof...(DefaultArgs)) {
			[]<bool flag = false>() { static_assert(flag, "Not enough default values"); }
			();
		}
		if constexpr (max_size() - min_size() < sizeof...(DefaultArgs)) {
			[]<bool flag = false>() { static_assert(flag, "Too many default values"); }
			();
		}

		if (kwargs != nullptr && !kwargs->map().empty()) {
			return Err(type_error("{} takes no keyword arguments", function_name));
		}

		if constexpr (max_size() - min_size() == 0) {
			if (args->size() != min_size()) {
				if constexpr (min_size() == 1) {
					return Err(type_error(
						"{} takes exactly one argument ({} given)", function_name, args->size()));
				} else {
					return Err(type_error("{} takes exactly {} arguments ({} given)",
						function_name,
						min_size(),
						args->size()));
				}
			}
		}

		if (args->size() < min_size()) {
			return Err(type_error(
				"function takes at least {} arguments ({} given)'", min_size(), args->size()));
		} else if (args->size() > max_size()) {
			return Err(type_error(
				"function takes at most {} argument ({} given)'", max_size(), args->size()));
		}

		std::tuple<ArgTypes...> unpacked_args;
		auto result = unpack_tuple_helper<0>(args->elements(),
			function_name,
			min_size,
			max_size,
			unpacked_args,
			std::forward<DefaultArgs>(default_values)...);

		if (result.is_err()) return Err(result.unwrap_err());
		return Ok(unpacked_args);
	}
};

}// namespace py