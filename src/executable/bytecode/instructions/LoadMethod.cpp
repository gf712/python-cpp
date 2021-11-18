#include "LoadMethod.hpp"
#include "runtime/PyModule.hpp"

void LoadMethod::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto this_value = vm.reg(m_value_source);
	spdlog::debug("This object: {}",
		std::visit([](const auto &val) { return PyObject::from(val)->to_string(); }, this_value));
	if (auto *this_obj = std::get_if<PyObject *>(&this_value)) {
		if (auto module = as<PyModule>(*this_obj)) {
			ASSERT(module->symbol_table().contains(PyString::create(m_method_name)))
			auto func_maybe = module->symbol_table().at(PyString::create(m_method_name));
			if (auto func = std::get<PyObject *>(func_maybe)) {
				ASSERT(func->type() == PyObjectType::PY_FUNCTION
					   || func->type() == PyObjectType::PY_NATIVE_FUNCTION)
				vm.reg(m_destination) = std::move(func);
			} else {
				// not a callable, raise exception
				TODO()
			}
		} else {
			auto maybe_method = (*this_obj)->get(m_method_name, interpreter);
			ASSERT(maybe_method)
			ASSERT(maybe_method->type() == PyObjectType::PY_FUNCTION
				   || maybe_method->type() == PyObjectType::PY_NATIVE_FUNCTION
				   || maybe_method->type() == PyObjectType::PY_METHOD_DESCRIPTOR
				   || maybe_method->type() == PyObjectType::PY_BOUND_METHOD)
			vm.reg(m_destination) = std::move(maybe_method);
		}
	} else {
		TODO();
	}
}
