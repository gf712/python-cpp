#include "runtime/forward.hpp"


// TODO: all of these concepts should check the return type
template<typename T> concept HasCreate = requires(T *self, PyTuple *args, PyDict *kwargs)
{
	T::create(self, args, kwargs);
};

template<typename T> concept HasCall = requires(T *self, PyTuple *args, PyDict *kwargs)
{
	self->__call__(args, kwargs);
};

template<typename T> concept HasNew = requires(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	T::__new__(type, args, kwargs);
};

template<typename T> concept HasInit = requires(T *self, PyTuple *args, PyDict *kwargs)
{
	self->__init__(args, kwargs);
};

template<typename T> concept HasGetAttro = requires(const T *self, PyObject *attr)
{
	self->__getattribute__(attr);
};

template<typename T> concept HasGet = requires(const T *self, PyObject *instance, PyObject *owner)
{
	self->__get__(instance, owner);
};

template<typename T> concept HasSetAttro = requires(T *self, PyObject *attr, PyObject *value)
{
	self->__setattribute__(attr, value);
};

template<typename T> concept HasHash = requires(const T *self) { self->__hash__(); };

template<typename T> concept HasDelete = requires(T *self) { self->~T(); };

template<typename T> concept HasRepr = requires(const T *obj) { obj->__repr__(); };

template<typename T> concept HasStr = requires(const T *obj) { obj->__str__(); };

template<typename T> concept HasBytes = requires(const T *obj) { obj->__bytes__(); };

template<typename T> concept HasFormat = requires(const T *obj) { obj->__format__(); };
template<typename T> concept HasLength = requires(const T *obj) { obj->__len__(); };

template<typename T> concept HasLt = requires(const T *obj, const PyObject *other)
{
	obj->__lt__(other);
};
template<typename T> concept HasLe = requires(const T *obj, const PyObject *other)
{
	obj->__le__(other);
};
template<typename T> concept HasEq = requires(const T *obj, const PyObject *other)
{
	obj->__eq__(other);
};
template<typename T> concept HasNe = requires(const T *obj, const PyObject *other)
{
	obj->__ne__(other);
};
template<typename T> concept HasGt = requires(const T *obj, const PyObject *other)
{
	obj->__gt__(other);
};
template<typename T> concept HasGe = requires(const T *obj, const PyObject *other)
{
	obj->__ge__(other);
};

template<typename T> concept HasBool = requires(const T *obj) { obj->__bool__(); };

template<typename T> concept HasIter = requires(const T *obj) { obj->__iter__(); };
template<typename T> concept HasRichCompare = requires(const T *obj, const PyObject *other)
{
	obj->__richcompare__(other);
};

template<typename T> concept HasNext = requires(T *obj) { obj->__next__(); };

template<typename T> concept HasAdd = requires(const T *obj, const PyObject *other)
{
	obj->__add__(other);
};

template<typename T> concept HasSub = requires(const T *obj, const PyObject *other)
{
	obj->__sub__(other);
};

template<typename T> concept HasMul = requires(const T *obj, const PyObject *other)
{
	obj->__mul__(other);
};

template<typename T> concept HasExp = requires(const T *obj, const PyObject *other)
{
	obj->__exp__(other);
};

template<typename T> concept HasLshift = requires(const T *obj, const PyObject *other)
{
	obj->__lshift__(other);
};

template<typename T> concept HasModulo = requires(const T *obj, const PyObject *other)
{
	obj->__mod__(other);
};

template<typename T> concept HasAbs = requires(const T *obj) { obj->__abs__(); };

template<typename T> concept HasNeg = requires(const T *obj) { obj->__neg__(); };

template<typename T> concept HasPos = requires(const T *obj) { obj->__pos__(); };

template<typename T> concept HasInvert = requires(const T *obj) { obj->__invert__(); };
