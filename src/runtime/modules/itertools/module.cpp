#include "../Modules.hpp"
#include "Chain.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyType.hpp"

namespace py {

namespace {
	static constexpr std::string_view kDoc = "";
}

PyModule *itertools_module()
{
	auto symbols = PyDict::create().unwrap();
	auto name = PyString::create("itertools").unwrap();
	auto doc = PyString::create(std::string{ kDoc }).unwrap();

	auto *module = PyModule::create(symbols, name, doc).unwrap();
	module->add_symbol(PyString::create("chain").unwrap(), itertools::Chain::register_type(module));

	return module;
}
}// namespace py
