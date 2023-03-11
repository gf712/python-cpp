#include "ImportStar.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/AttributeError.hpp"
#include "runtime/IndexError.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyType.hpp"
#include "runtime/TypeError.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> ImportStar::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	auto module_ = vm.reg(m_src);
	ASSERT(std::holds_alternative<PyObject *>(module_));
	auto *obj = std::get<PyObject *>(module_);
	ASSERT(as<PyModule>(obj));
	auto *module_obj = as<PyModule>(obj);
	auto *symbol_table = module_obj->symbol_table();
	if (const auto it = symbol_table->map().find(String{ "__all__" });
		it != symbol_table->map().end()) {
		auto all_ = PyObject::from(it->second);
		if (all_.is_err()) { return all_; }
		auto all_sequence_ = all_.unwrap()->as_sequence();
		if (all_sequence_.is_err()) { return Err(all_sequence_.unwrap_err()); }
		auto all_sequence = all_sequence_.unwrap();
		for (int64_t i = 0;; ++i) {
			auto el = all_sequence.getitem(i);
			if (el.is_err()) {
				if (el.unwrap_err()->type()->issubclass(IndexError::class_type())) { break; }
				return Err(el.unwrap_err());
			}
			auto name = as<PyString>(el.unwrap());
			if (!name) {
				return Err(type_error("Item in {}.__all__ must be str, not {}",
					module_obj->name()->value(),
					el.unwrap()->type()->name()));
			}
			auto it = symbol_table->map().find(name);
			if (it == symbol_table->map().end()) {
				return Err(attribute_error("module '{}' has no attribute '{}'",
					module_obj->name()->value(),
					name->value()));
			}
			if (auto r = interpreter.store_object(name->value(), it->second); r.is_err()) {
				return Err(r.unwrap_err());
			}
		}
	} else {
		for (const auto &[key, value] : symbol_table->map()) {
			const auto &k = key;
			const auto key_str = [key = k]() -> std::string {
				if (std::holds_alternative<String>(key)) {
					return std::get<String>(key).s;
				} else if (std::holds_alternative<PyObject *>(key)) {
					ASSERT(as<PyString>(std::get<PyObject *>(key)));
					return as<PyString>(std::get<PyObject *>(key))->value();
				} else {
					TODO();
				}
			}();

			ASSERT(!key_str.empty());
			if (key_str[0] != '_') {
				if (auto r = interpreter.store_object(key_str, value); r.is_err()) {
					return Err(r.unwrap_err());
				}
			}
		}
	}
	return Ok(py_none());
}

std::vector<uint8_t> ImportStar::serialize() const
{
	return {
		IMPORT_STAR,
		m_src,
	};
}
