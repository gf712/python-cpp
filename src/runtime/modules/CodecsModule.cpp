#include "Modules.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/LookupError.hpp"
#include "runtime/PyArgParser.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/Value.hpp"
#include "runtime/types/builtin.hpp"
#include "vm/VM.hpp"

#include <algorithm>
#include <cctype>


namespace py {
namespace detail {
	namespace {
		PyResult<PyObject *> lookup_error(PyTuple *args, PyDict *kwargs)
		{
			auto result = PyArgsParser<PyString *>::unpack_tuple(args,
				kwargs,
				"lookup_errors",
				std::integral_constant<size_t, 1>{},
				std::integral_constant<size_t, 1>{});

			if (result.is_err()) { return Err(result.unwrap_err()); }
			auto [error] = result.unwrap();

			if (auto it =
					VirtualMachine::the().interpreter().codec_error_registry()->map().find(error);
				it != VirtualMachine::the().interpreter().codec_error_registry()->map().end()) {
				return PyObject::from(it->second);
			}

			return Err(py::lookup_error("unknown error handler name '{}'", error->value()));
		}

		PyResult<PyObject *> lookup(PyTuple *args, PyDict *kwargs)
		{
			auto parse_result = PyArgsParser<PyString *>::unpack_tuple(args,
				kwargs,
				"lookup",
				std::integral_constant<size_t, 1>{},
				std::integral_constant<size_t, 1>{});

			if (parse_result.is_err()) { return Err(parse_result.unwrap_err()); }
			auto [encoding] = parse_result.unwrap();

			std::string normalized_encoding;
			normalized_encoding.reserve(encoding->value().size());
			std::transform(encoding->value().begin(),
				encoding->value().end(),
				std::back_inserter(normalized_encoding),
				[](char el) {
					if (isspace(el) || el == '-') { return '_'; }
					return static_cast<char>(std::tolower(el));
				});
			auto py_normalized_encoding = PyString::create(std::move(normalized_encoding));
			if (py_normalized_encoding.is_err()) {
				return Err(py_normalized_encoding.unwrap_err());
			}
			if (auto it = VirtualMachine::the().interpreter().codec_search_path_cache()->map().find(
					py_normalized_encoding.unwrap());
				it != VirtualMachine::the().interpreter().codec_search_path_cache()->map().end()) {
				return PyObject::from(it->second);
			}
			auto args_ = PyTuple::create(py_normalized_encoding.unwrap());
			PyObject *result = nullptr;
			if (args_.is_err()) { return Err(args_.unwrap_err()); }
			auto *codec_args = args_.unwrap();
			for (const auto &el :
				VirtualMachine::the().interpreter().codec_search_path()->elements()) {
				auto func_ = PyObject::from(el);
				if (func_.is_err()) { return func_; }
				auto *func = func_.unwrap();
				auto result_ = func->call(codec_args, nullptr);
				if (result_.is_err()) { return result_; }
				if (result_.unwrap() == py::py_none()) { continue; }
				if (!result_.unwrap()->type()->issubclass(py::types::tuple())
					|| static_cast<const PyTuple &>(*result_.unwrap()).size() != 4) {
					return Err(type_error("codec search functions must return 4-tuples"));
				}
				result = result_.unwrap();
				break;
			}

			if (!result) {
				return Err(py::lookup_error("unknown encoding: {}", encoding->value()));
			}

			VirtualMachine::the().interpreter().codec_search_path_cache()->insert(
				py_normalized_encoding.unwrap(), result);

			return Ok(result);
		}
	}// namespace
}// namespace detail

PyModule *codecs_module()
{
	auto *s_codecs_module = PyModule::create(PyDict::create().unwrap(),
		PyString::create("_codecs").unwrap(),
		PyString::create("").unwrap())
								.unwrap();

	s_codecs_module->add_symbol(PyString::create("lookup_error").unwrap(),
		PyNativeFunction::create("lookup_error", detail::lookup_error).unwrap());

	s_codecs_module->add_symbol(PyString::create("lookup").unwrap(),
		PyNativeFunction::create("lookup", detail::lookup).unwrap());

	return s_codecs_module;
}
}// namespace py