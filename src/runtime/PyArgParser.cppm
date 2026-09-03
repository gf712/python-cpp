module;
#include "core.hpp"

export module py.runtime:arg_parser;
import :type;
import :type_error;
import :vm;
import :dict;
import :integer;
import :object;
import :tuple;
import :value;
import py.memory;
import std;

// Moved from utilities.hpp, which pulls spdlog and so cannot enter a GMF.
template<typename T>
concept Integral = std::is_integral_v<T>;

export template<Integral Target, Integral Source> bool fits_in(Source value)
{
	constexpr bool source_is_signed = std::is_signed<Source>::value;
	constexpr bool target_is_signed = std::is_signed<Target>::value;

	// Case 1: Source is signed, Target is unsigned
	if constexpr (source_is_signed && !target_is_signed) {
		if (value < 0) { return false; }
		// Now we know value is non-negative, safe to cast and compare
		return static_cast<std::make_unsigned_t<Source>>(value)
			   <= std::numeric_limits<Target>::max();
	}
	// Case 2: Source is unsigned, Target is signed
	else if constexpr (!source_is_signed && target_is_signed) {
		return value
			   <= static_cast<std::make_unsigned_t<Target>>(std::numeric_limits<Target>::max());
	} else {
		// Case 3: Both signed or both unsigned
		return value >= std::numeric_limits<Target>::min()
			   && value <= std::numeric_limits<Target>::max();
	}
}
export namespace py {

template<typename... ArgTypes> struct PyArgsParser
{
  private:
	template<std::size_t Idx,
		std::size_t MinSize,
		std::size_t MaxSize,
		typename ResultType,
		typename... DefaultArgs>
	static constexpr PyResult<std::monostate> unpack_tuple_helper(const std::vector<Value> &args,
		std::string_view function_name,
		std::integral_constant<std::size_t, MinSize> min_size,
		std::integral_constant<std::size_t, MaxSize> max_size,
		ResultType &result,
		DefaultArgs &&...default_args)
	{
		using ExpectedType = std::tuple_element_t<Idx, ResultType>;
		if (args.size() > Idx) {
			if constexpr (std::is_base_of_v<PyObject,
							  std::remove_pointer_t<std::remove_cv_t<ExpectedType>>>) {
				using PyObjectType = std::remove_pointer_t<std::remove_cv_t<ExpectedType>>;
				const auto &arg = PyObject::from(args[Idx]);
				if (arg.is_err()) return Err(arg.unwrap_err());
				if constexpr (std::is_same_v<PyObject, PyObjectType>) {
					std::get<Idx>(result) = arg.unwrap();
				} else {
					if (!as<PyObjectType>(arg.unwrap())) return Err(type_error("Unexpected type"));
					std::get<Idx>(result) = as<PyObjectType>(arg.unwrap());
				}
			} else if constexpr (std::is_same_v<bool,
									 std::remove_pointer_t<std::remove_cv_t<ExpectedType>>>) {
				if (auto bool_arg = truthy(args[Idx], VirtualMachine::the().interpreter());
					bool_arg.is_ok()) {
					std::get<Idx>(result) = bool_arg.unwrap();
				} else {
					return Err(bool_arg.unwrap_err());
				}
			} else if constexpr (std::is_same_v<bool,
									 std::remove_pointer_t<std::remove_cv_t<ExpectedType>>>) {
				TODO();
			} else if constexpr (std::is_integral_v<ExpectedType>) {
				auto int_obj = PyObject::from(args[Idx]);
				if (int_obj.is_err()) { return Err(int_obj.unwrap_err()); }
				if (!as<PyInteger>(int_obj.unwrap())) {
					return Err(type_error("'{}' object cannot be interpreted as an integer",
						int_obj.unwrap()->type()->name()));
				} else {
					auto value = as<PyInteger>(int_obj.unwrap())->as_i64();
					static_assert(sizeof(ExpectedType) <= 8);
					if (!fits_in<ExpectedType>(value)) {
						return Err(type_error("{} not within range ({}, {})",
							value,
							std::numeric_limits<ExpectedType>::min(),
							std::numeric_limits<ExpectedType>::max()));
					}
					std::get<Idx>(result) = static_cast<ExpectedType>(value);
				}
			} else {
				[]<bool flag = false>() {
					static_assert(flag, "unsupported Python to native conversion");
				}();
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
	}

  public:
	template<std::size_t MinSize, std::size_t MaxSize, typename... DefaultArgs>
	static constexpr PyResult<std::tuple<ArgTypes...>> unpack_tuple(PyTuple *args,
		PyDict *kwargs,
		std::string_view function_name,
		std::integral_constant<std::size_t, MinSize> min_size,
		std::integral_constant<std::size_t, MaxSize> max_size,
		DefaultArgs &&...default_values)
	{
		if constexpr (max_size() - min_size() > sizeof...(DefaultArgs)) {
			[]<bool flag = false>() { static_assert(flag, "Not enough default values"); }();
		}
		if constexpr (max_size() - min_size() < sizeof...(DefaultArgs)) {
			[]<bool flag = false>() { static_assert(flag, "Too many default values"); }();
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
