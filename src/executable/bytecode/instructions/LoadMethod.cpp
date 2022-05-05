#include "LoadMethod.hpp"
#include "runtime/AttributeError.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyType.hpp"
#include "runtime/types/builtin.hpp"

using namespace py;

PyResult<Value> LoadMethod::execute(VirtualMachine &vm, Interpreter &) const
{
	auto this_value = vm.reg(m_value_source);
	spdlog::debug("This object: {}",
		std::visit(
			[](const auto &val) {
				auto obj = PyObject::from(val);
				ASSERT(obj.is_ok())
				return obj.unwrap()->to_string();
			},
			this_value));
	auto this_obj_ = PyObject::from(this_value);
	if (this_obj_.is_err()) { return Err(this_obj_.unwrap_err()); }
	auto *this_obj = this_obj_.unwrap();
	if (auto module = as<PyModule>(this_obj)) {
		auto name = PyString::create(m_method_name);
		if (name.is_err()) { return Err(name.unwrap_err()); }
		ASSERT(module->symbol_table().contains(name.unwrap()))
		auto func_maybe = module->symbol_table().at(name.unwrap());
		if (auto func = std::get<PyObject *>(func_maybe)) {
			ASSERT(func->type() == function() || func->type() == native_function())
			if (!func->is_callable()) {
				return Err(attribute_error("object '{}' is not callable", func->name()));
			}
			vm.reg(m_destination) = func;
			return Ok(Value{ func });
		} else {
			return Err(attribute_error("object '{}' is not callable", this_obj->name()));
		}
	} else {
		auto name = PyString::create(m_method_name);
		if (name.is_err()) { return Err(name.unwrap_err()); }
		auto maybe_method = this_obj->get_method(name.unwrap());
		if (maybe_method.is_err()) {
			// FIXME: maybe check here if there is an active exception?
			return Err(attribute_error(
				"object '{}' has no attribute '{}'", this_obj->name(), m_method_name));
		}
		ASSERT(maybe_method.is_ok())
		ASSERT(maybe_method.unwrap()->is_callable())
		if (auto *method = maybe_method.unwrap(); !method->is_callable()) {
			return Err(attribute_error("object '{}' is not callable", method->name()));
		} else {
			vm.reg(m_destination) = method;
		}
		return Ok(Value{ maybe_method.unwrap() });
	}
}

std::vector<uint8_t> LoadMethod::serialize() const
{
	TODO();
	return {
		LOAD_METHOD,
		m_destination,
		m_value_source,
	};
}
