#include "../Modules.hpp"
#include "runtime/PyDict.hpp"

namespace py {

namespace {
	static constexpr std::string_view kDoc =
		R"(This module provides access to the mathematical functions
defined by the C standard.)";
}

PyModule *math_module()
{
	auto symbols = PyDict::create().unwrap();
	auto name = PyString::create("math").unwrap();
	auto doc = PyString::create(std::string{ kDoc }).unwrap();

	auto *module = PyModule::create(symbols, name, doc).unwrap();

	return module;
}
}// namespace py
