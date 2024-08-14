#include "../Modules.hpp"
#include "DefaultDict.hpp"
#include "Deque.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyType.hpp"

namespace py {

namespace {
	static constexpr std::string_view kDoc = "";
}

PyModule *collections_module()
{
	auto symbols = PyDict::create().unwrap();
	auto name = PyString::create("_collections").unwrap();
	auto doc = PyString::create(std::string{ kDoc }).unwrap();

	auto *module = PyModule::create(symbols, name, doc).unwrap();
	module->add_symbol(
		PyString::create("defaultdict").unwrap(), collections::DefaultDict::register_type(module));
	module->add_symbol(
		PyString::create("deque").unwrap(), collections::Deque::register_type(module));

	return module;
}
}// namespace py
