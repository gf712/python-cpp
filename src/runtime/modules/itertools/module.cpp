#include "../Modules.hpp"
#include "Chain.hpp"
#include "ISlice.hpp"
#include "Repeat.hpp"
#include "StarMap.hpp"
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
	module->add_symbol(
		PyString::create("islice").unwrap(), itertools::ISlice::register_type(module));
	module->add_symbol(
		PyString::create("repeat").unwrap(), itertools::Repeat::register_type(module));
	module->add_symbol(
		PyString::create("starmap").unwrap(), itertools::StarMap::register_type(module));

	return module;
}
}// namespace py
