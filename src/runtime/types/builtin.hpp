#pragma once

#include "runtime/forward.hpp"
#include <memory>

class BuiltinTypes
{
	std::unique_ptr<TypePrototype> m_type;
	std::unique_ptr<TypePrototype> m_str;
	std::unique_ptr<TypePrototype> m_bool;
	std::unique_ptr<TypePrototype> m_bytes;
	std::unique_ptr<TypePrototype> m_ellipsis;
	std::unique_ptr<TypePrototype> m_float;
	std::unique_ptr<TypePrototype> m_integer;
	std::unique_ptr<TypePrototype> m_none;
	std::unique_ptr<TypePrototype> m_module;
	std::unique_ptr<TypePrototype> m_object;

	std::unique_ptr<TypePrototype> m_function;
	std::unique_ptr<TypePrototype> m_native_function;
	std::unique_ptr<TypePrototype> m_code;

	std::unique_ptr<TypePrototype> m_dict;
	std::unique_ptr<TypePrototype> m_dict_items;
	std::unique_ptr<TypePrototype> m_dict_items_iterator;

	std::unique_ptr<TypePrototype> m_list;
	std::unique_ptr<TypePrototype> m_list_iterator;

	std::unique_ptr<TypePrototype> m_tuple;
	std::unique_ptr<TypePrototype> m_tuple_iterator;

	std::unique_ptr<TypePrototype> m_range;
	std::unique_ptr<TypePrototype> m_range_iterator;

	std::unique_ptr<TypePrototype> m_builtin_method;
	std::unique_ptr<TypePrototype> m_slot_wrapper;
	std::unique_ptr<TypePrototype> m_bound_method;
	std::unique_ptr<TypePrototype> m_method_wrapper;
	std::unique_ptr<TypePrototype> m_static_method;

	std::unique_ptr<TypePrototype> m_base_exception;

	BuiltinTypes();

  public:
	static BuiltinTypes &the()
	{
		static auto instance = BuiltinTypes();
		return instance;
	}

	TypePrototype &type() const { return *m_type; }
	TypePrototype &bool_() const { return *m_bool; }
	TypePrototype &bytes() const { return *m_bytes; }
	TypePrototype &ellipsis() const { return *m_ellipsis; }
	TypePrototype &str() const { return *m_str; }
	TypePrototype &float_() const { return *m_float; }
	TypePrototype &integer() const { return *m_integer; }
	TypePrototype &none() const { return *m_none; }
	TypePrototype &module() const { return *m_module; }
	TypePrototype &object() const { return *m_object; }

	TypePrototype &dict() const { return *m_dict; }
	TypePrototype &dict_items() const { return *m_dict_items; }
	TypePrototype &dict_items_iterator() const { return *m_dict_items_iterator; }

	TypePrototype &list() const { return *m_list; }
	TypePrototype &list_iterator() const { return *m_list_iterator; }

	TypePrototype &tuple() const { return *m_tuple; }
	TypePrototype &tuple_iterator() const { return *m_tuple_iterator; }

	TypePrototype &range() const { return *m_range; }
	TypePrototype &range_iterator() const { return *m_range_iterator; }

	TypePrototype &function() const { return *m_function; }
	TypePrototype &native_function() const { return *m_native_function; }
	TypePrototype &code() const { return *m_code; }

	TypePrototype &builtin_method() const { return *m_builtin_method; }
	TypePrototype &slot_wrapper() const { return *m_slot_wrapper; }
	TypePrototype &bound_method() const { return *m_bound_method; }
	TypePrototype &method_wrapper() const { return *m_method_wrapper; }
	TypePrototype &static_method() const { return *m_static_method; }

	TypePrototype &base_exception() const { return *m_base_exception; }
};

PyType *type();
PyType *bool_();
PyType *bytes();
PyType *ellipsis();
PyType *str();
PyType *float_();
PyType *integer();
PyType *none();
PyType *module();
PyType *object();
PyType *dict();
PyType *dict_items();
PyType *dict_items_iterator();
PyType *list();
PyType *list_iterator();
PyType *tuple();
PyType *tuple_iterator();
PyType *range();
PyType *range_iterator();
PyType *function();
PyType *native_function();
PyType *code();
PyType *builtin_method();
PyType *slot_wrapper();
PyType *bound_method();
PyType *method_wrapper();
PyType *static_method();