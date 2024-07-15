#pragma once

#include "runtime/forward.hpp"

#include <concepts>

namespace py {

template<typename _Interface, typename _Interfacing> struct Interface : public _Interface
{
	using _InterfaceT = _Interface;
	using _InterfacingT = _Interfacing;
	template<typename... Args> Interface(Args &&...args) : _Interface(std::forward<Args>(args)...)
	{}
};

namespace concepts {
	template<typename T>
	concept HasInterface = std::is_base_of_v<Interface<typename T::_InterfaceT, T>, T>;

	// TODO: all of these concepts should check the return type
	template<typename T>
	concept HasCreate = requires(T *self, PyTuple *args, PyDict *kwargs)
	{
		T::create(self, args, kwargs);
	};

	template<typename T>
	concept HasCall = requires(T *self, PyTuple *args, PyDict *kwargs)
	{
		self->__call__(args, kwargs);
	};

	template<typename T>
	concept HasNew = requires(const PyType *type, PyTuple *args, PyDict *kwargs)
	{
		T::__new__(type, args, kwargs);
	};

	template<typename T>
	concept HasInit = requires(T *self, PyTuple *args, PyDict *kwargs)
	{
		self->__init__(args, kwargs);
	};

	template<typename T>
	concept HasDoc = requires
	{
		{ T::__doc__ } -> std::convertible_to<std::string_view>;
	};

	template<typename T>
	concept HasGetAttro = requires(const T *self, PyObject *attr)
	{
		self->__getattribute__(attr);
	};

	template<typename T>
	concept HasGet = requires(const T *self, PyObject *instance, PyObject *owner)
	{
		self->__get__(instance, owner);
	};

	template<typename T>
	concept HasSet = requires(T *self, PyObject *attribute, PyObject *value)
	{
		self->__set__(attribute, value);
	};

	template<typename T>
	concept HasSetAttro = requires(T *self, PyObject *attr, PyObject *value)
	{
		self->__setattribute__(attr, value);
	};

	template<typename T>
	concept HasHash = requires(const T *self)
	{
		self->__hash__();
	};

	template<typename T>
	concept HasDelete = requires(T *self)
	{
		self->~T();
	};

	template<typename T>
	concept HasRepr = requires(const T *obj)
	{
		obj->__repr__();
	};

	template<typename T>
	concept HasStr = requires(T *obj)
	{
		obj->__str__();
	};

	template<typename T>
	concept HasBytes = requires(const T *obj)
	{
		obj->__bytes__();
	};

	template<typename T>
	concept HasFormat = requires(const T *obj)
	{
		obj->__format__();
	};

	// mapping methods
	template<typename T>
	concept HasLength = requires(const T *obj)
	{
		obj->__len__();
	};

	template<typename T>
	concept HasGetItem = requires(T *obj, PyObject *name)
	{
		obj->__getitem__(name);
	};
	template<typename T>
	concept HasSetItem = requires(T *obj, PyObject *name, PyObject *value)
	{
		obj->__setitem__(name, value);
	};
	template<typename T>
	concept HasDelItem = requires(T *obj, PyObject *name)
	{
		obj->__delitem__(name);
	};

	// sequence methods
	template<typename T>
	concept HasContains = requires(T *obj, PyObject *value)
	{
		obj->__contains__(value);
	};

	template<typename T>
	concept HasSequenceGetItem = requires(T *obj, int64_t index)
	{
		obj->__getitem__(index);
	};
	template<typename T>
	concept HasSequenceSetItem = requires(T *obj, int64_t index, PyObject *value)
	{
		obj->__setitem__(index, value);
	};
	template<typename T>
	concept HasSequenceDelItem = requires(T *obj, int64_t index)
	{
		obj->__delitem__(index);
	};

	template<typename T>
	concept HasLt = requires(const T *obj, const PyObject *other)
	{
		obj->__lt__(other);
	};
	template<typename T>
	concept HasLe = requires(const T *obj, const PyObject *other)
	{
		obj->__le__(other);
	};
	template<typename T>
	concept HasEq = requires(const T *obj, const PyObject *other)
	{
		obj->__eq__(other);
	};
	template<typename T>
	concept HasNe = requires(const T *obj, const PyObject *other)
	{
		obj->__ne__(other);
	};
	template<typename T>
	concept HasGt = requires(const T *obj, const PyObject *other)
	{
		obj->__gt__(other);
	};
	template<typename T>
	concept HasGe = requires(const T *obj, const PyObject *other)
	{
		obj->__ge__(other);
	};

	template<typename T>
	concept HasAnd = requires(T *self, PyObject *other)
	{
		self->__and__(other);
	};

	template<typename T>
	concept HasOr = requires(T *self, PyObject *other)
	{
		self->__or__(other);
	};

	template<typename T>
	concept HasBool = requires(const T *obj)
	{
		obj->__bool__();
	};

	template<typename T>
	concept HasIter = requires(const T *obj)
	{
		obj->__iter__();
	};
	template<typename T>
	concept HasRichCompare = requires(const T *obj, const PyObject *other)
	{
		obj->__richcompare__(other);
	};

	template<typename T>
	concept HasNext = requires(T *obj)
	{
		obj->__next__();
	};

	namespace detail {
		template<typename T> constexpr bool has_add()
		{
			if constexpr (HasInterface<T>) { return true; }
			return std::is_same_v<decltype(&T::__add__),
				PyResult<PyObject *> (T::*)(const PyObject *) const>;
		}
	}// namespace detail

	template<typename T>
	concept HasAdd = requires(const T *obj, const PyObject *other)
	{
		{ obj->__add__(other) };
		requires detail::has_add<T>();
	};

	template<typename T>
	concept HasSub = requires(const T *obj, const PyObject *other)
	{
		obj->__sub__(other);
	};

	template<typename T>
	concept HasMul = requires(const T *obj, const PyObject *other)
	{
		obj->__mul__(other);
	};

	template<typename T>
	concept HasPow = requires(const T *obj, const PyObject *other, const PyObject *modulo)
	{
		obj->__pow__(other, modulo);
	};

	template<typename T>
	concept HasFloorDiv = requires(T *obj, PyObject *other)
	{
		obj->__floordiv__(other);
	};

	template<typename T>
	concept HasTrueDiv = requires(T *obj, PyObject *other)
	{
		obj->__truediv__(other);
	};

	template<typename T>
	concept HasLshift = requires(const T *obj, const PyObject *other)
	{
		obj->__lshift__(other);
	};

	template<typename T>
	concept HasRshift = requires(const T *obj, const PyObject *other)
	{
		obj->__rshift__(other);
	};

	template<typename T>
	concept HasModulo = requires(const T *obj, const PyObject *other)
	{
		obj->__mod__(other);
	};

	template<typename T>
	concept HasAbs = requires(const T *obj)
	{
		obj->__abs__();
	};

	template<typename T>
	concept HasNeg = requires(const T *obj)
	{
		obj->__neg__();
	};

	template<typename T>
	concept HasPos = requires(const T *obj)
	{
		obj->__pos__();
	};

	template<typename T>
	concept HasInvert = requires(const T *obj)
	{
		obj->__invert__();
	};

}// namespace concepts
}// namespace py
