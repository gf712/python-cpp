#include "runtime/forward.hpp"


template<typename T> concept HasNew = requires(PyTuple *args, PyDict *kwargs)
{
	T::create(args, kwargs);
};

template<typename T> concept HasInit = requires(T *self, PyTuple *args, PyDict *kwargs)
{
	self->__init__(args, kwargs);
};
#
template<typename T> concept HasDelete = requires(T *self) { self->~T(); };

template<typename T> concept HasRepr = requires(T *obj) { obj->__repr__(); };

template<typename T> concept HasStr = requires(T *obj) { obj->__str__(); };

template<typename T> concept HasBytes = requires(T *obj) { obj->__bytes__(); };

template<typename T> concept HasFormat = requires(T *obj) { obj->__format__(); };

template<typename T> concept HasLt = requires(T *obj, PyObject *other) { obj->__lt__(other); };
template<typename T> concept HasLe = requires(T *obj, PyObject *other) { obj->__le__(other); };
template<typename T> concept HasEq = requires(T *obj, PyObject *other) { obj->__eq__(other); };
template<typename T> concept HasNe = requires(T *obj, PyObject *other) { obj->__ne__(other); };
template<typename T> concept HasGt = requires(T *obj, PyObject *other) { obj->__gt__(other); };
template<typename T> concept HasGe = requires(T *obj, PyObject *other) { obj->__ge__(other); };

template<typename T> concept HasBool = requires(T *obj) { obj->__bool__(); };

template<typename T> concept HasAdd = requires(T *obj, const PyObject *other)
{
	obj->add_impl(other);
};
template<typename T> concept HasIter = requires(T *obj) { obj->__iter__(); };
template<typename T> concept HasRichCompare = requires(T *obj, PyObject *other)
{
	obj->__richcompare__(other);
};
