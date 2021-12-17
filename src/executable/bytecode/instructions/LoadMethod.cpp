#include "LoadMethod.hpp"
#include "runtime/AttributeError.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyType.hpp"
#include "runtime/types/builtin.hpp"


void LoadMethod::execute(VirtualMachine &vm, Interpreter &) const
{
	auto this_value = vm.reg(m_value_source);
	spdlog::debug("This object: {}",
		std::visit([](const auto &val) { return PyObject::from(val)->to_string(); }, this_value));
	auto *this_obj = PyObject::from(this_value);
	if (auto module = as<PyModule>(this_obj)) {
		PyString *name = PyString::create(m_method_name);
		ASSERT(module->symbol_table().contains(name))
		auto func_maybe = module->symbol_table().at(name);
		if (auto func = std::get<PyObject *>(func_maybe)) {
			ASSERT(func->type() == function() || func->type() == native_function())
			vm.reg(m_destination) = std::move(func);
		} else {
			// not a callable, raise exception
			TODO();
		}
	} else {
		auto maybe_method = this_obj->get_method(PyString::create(m_method_name));
		if (!maybe_method) {
			// FIXME: maybe check here if there is an active exception? 
			// interpreter.raise_exception(attribute_error(
			// 	"object '{}' has no attribute '{}'", this_obj->name(), m_method_name));
			return;
		}
		ASSERT(maybe_method)
		ASSERT(maybe_method->is_callable())
		vm.reg(m_destination) = maybe_method;
	}
}
