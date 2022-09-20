#pragma once

#include "runtime/forward.hpp"
#include <functional>
#include <memory>
#include <variant>

namespace py {
class BuiltinTypes
{
	using Type = std::variant<std::function<std::unique_ptr<TypePrototype>()>,
		std::unique_ptr<TypePrototype>>;
	mutable Type m_type;
	mutable Type m_super;

	mutable Type m_str;
	mutable Type m_str_iterator;
	mutable Type m_integer;
	mutable Type m_bool;
	mutable Type m_bytes;
	mutable Type m_ellipsis;
	mutable Type m_float;
	mutable Type m_none;
	mutable Type m_module;
	mutable Type m_object;

	mutable Type m_function;
	mutable Type m_native_function;
	mutable Type m_llvm_function;
	mutable Type m_code;
	mutable Type m_cell;

	mutable Type m_dict;
	mutable Type m_dict_items;
	mutable Type m_dict_items_iterator;
	mutable Type m_dict_keys;
	mutable Type m_dict_key_iterator;
	mutable Type m_dict_values;
	mutable Type m_dict_value_iterator;

	mutable Type m_list;
	mutable Type m_list_iterator;

	mutable Type m_tuple;
	mutable Type m_tuple_iterator;

	mutable Type m_set;
	mutable Type m_set_iterator;

	mutable Type m_range;
	mutable Type m_range_iterator;

	mutable Type m_slice;

	mutable Type m_builtin_method;
	mutable Type m_slot_wrapper;
	mutable Type m_bound_method;
	mutable Type m_method_wrapper;
	mutable Type m_classmethod_descriptor;
	mutable Type m_getset_descriptor;
	mutable Type m_static_method;
	mutable Type m_property;
	mutable Type m_classmethod;
	mutable Type m_member_descriptor;

	mutable Type m_base_exception;
	mutable Type m_traceback;

	mutable Type m_frame;

	mutable Type m_not_implemented;

	mutable Type m_namespace;

	mutable Type m_generator;
	mutable Type m_coroutine;
	mutable Type m_async_generator;

	mutable Type m_generic_alias;

	BuiltinTypes();

	TypePrototype &get_type(Type &t) const
	{
		if (std::holds_alternative<std::unique_ptr<TypePrototype>>(t)) {
			return *std::get<std::unique_ptr<TypePrototype>>(t);
		}
		t = std::get<std::function<std::unique_ptr<TypePrototype>()>>(t)();
		return *std::get<std::unique_ptr<TypePrototype>>(t);
	}

  public:
	static BuiltinTypes &the()
	{
		static auto instance = BuiltinTypes();
		return instance;
	}

	TypePrototype &type() const { return get_type(m_type); }
	TypePrototype &super() const { return get_type(m_super); }

	TypePrototype &bool_() const { return get_type(m_bool); }
	TypePrototype &bytes() const { return get_type(m_bytes); }
	TypePrototype &ellipsis() const { return get_type(m_ellipsis); }
	TypePrototype &str() const { return get_type(m_str); }
	TypePrototype &str_iterator() const { return get_type(m_str_iterator); }
	TypePrototype &float_() const { return get_type(m_float); }
	TypePrototype &integer() const { return get_type(m_integer); }
	TypePrototype &none() const { return get_type(m_none); }
	TypePrototype &module() const { return get_type(m_module); }
	TypePrototype &object() const { return get_type(m_object); }

	TypePrototype &dict() const { return get_type(m_dict); }
	TypePrototype &dict_items() const { return get_type(m_dict_items); }
	TypePrototype &dict_items_iterator() const { return get_type(m_dict_items_iterator); }
	TypePrototype &dict_keys() const { return get_type(m_dict_keys); }
	TypePrototype &dict_key_iterator() const { return get_type(m_dict_key_iterator); }
	TypePrototype &dict_values() const { return get_type(m_dict_values); }
	TypePrototype &dict_value_iterator() const { return get_type(m_dict_value_iterator); }

	TypePrototype &list() const { return get_type(m_list); }
	TypePrototype &list_iterator() const { return get_type(m_list_iterator); }

	TypePrototype &tuple() const { return get_type(m_tuple); }
	TypePrototype &tuple_iterator() const { return get_type(m_tuple_iterator); }

	TypePrototype &set() const { return get_type(m_set); }
	TypePrototype &set_iterator() const { return get_type(m_set_iterator); }

	TypePrototype &range() const { return get_type(m_range); }
	TypePrototype &range_iterator() const { return get_type(m_range_iterator); }

	TypePrototype &slice() const { return get_type(m_slice); }

	TypePrototype &function() const { return get_type(m_function); }
	TypePrototype &native_function() const { return get_type(m_native_function); }
	TypePrototype &llvm_function() const { return get_type(m_llvm_function); }
	TypePrototype &code() const { return get_type(m_code); }
	TypePrototype &cell() const { return get_type(m_cell); }

	TypePrototype &builtin_method() const { return get_type(m_builtin_method); }
	TypePrototype &slot_wrapper() const { return get_type(m_slot_wrapper); }
	TypePrototype &bound_method() const { return get_type(m_bound_method); }
	TypePrototype &method_wrapper() const { return get_type(m_method_wrapper); }
	TypePrototype &classmethod_descriptor() const { return get_type(m_classmethod_descriptor); }
	TypePrototype &getset_descriptor() const { return get_type(m_getset_descriptor); }
	TypePrototype &static_method() const { return get_type(m_static_method); }
	TypePrototype &property() const { return get_type(m_property); }
	TypePrototype &classmethod() const { return get_type(m_classmethod); }
	TypePrototype &member_descriptor() const { return get_type(m_member_descriptor); }

	TypePrototype &base_exception() const { return get_type(m_base_exception); }
	TypePrototype &traceback() const { return get_type(m_traceback); }

	TypePrototype &frame() const { return get_type(m_frame); }

	TypePrototype &not_implemented() const { return get_type(m_not_implemented); }

	TypePrototype &namespace_() const { return get_type(m_namespace); }

	TypePrototype &generator() const { return get_type(m_generator); }
	TypePrototype &coroutine() const { return get_type(m_coroutine); }
	TypePrototype &async_generator() const { return get_type(m_async_generator); }

	TypePrototype &generic_alias() const { return get_type(m_generic_alias); }
};

PyType *type();
PyType *super();
PyType *bool_();
PyType *bytes();
PyType *ellipsis();
PyType *str();
PyType *str_iterator();
PyType *float_();
PyType *integer();
PyType *none();
PyType *module();
PyType *object();
PyType *dict();
PyType *dict_items();
PyType *dict_items_iterator();
PyType *dict_keys();
PyType *dict_key_iterator();
PyType *dict_values();
PyType *dict_value_iterator();
PyType *list();
PyType *list_iterator();
PyType *tuple();
PyType *tuple_iterator();
PyType *set();
PyType *set_iterator();
PyType *range();
PyType *range_iterator();
PyType *slice();
PyType *function();
PyType *native_function();
PyType *llvm_function();
PyType *code();
PyType *cell();
PyType *builtin_method();
PyType *slot_wrapper();
PyType *bound_method();
PyType *method_wrapper();
PyType *classmethod_descriptor();
PyType *getset_descriptor();
PyType *static_method();
PyType *property();
PyType *classmethod();
PyType *member_descriptor();
PyType *traceback();
PyType *not_implemented();
PyType *frame();
PyType *namespace_();
PyType *generator();
PyType *coroutine();
PyType *async_generator();
PyType *generic_alias();

}// namespace py