#pragma once

namespace py {

class PyAsyncGenerator;
class PyCell;
class PyCode;
class PyCoroutine;
class PyDict;
class PyFloat;
class PyFrozenSet;
class PyFunction;
class PyFrame;
class PyGenerator;
class PyInteger;
class PyList;
class PyNativeFunction;
class PyNone;
class PyNumber;
class PyMethodDescriptor;
class PyModule;
class PyObject;
class PySet;
class PySlice;
class PySlotWrapper;
class PyString;
class PyTraceback;
class PyTuple;
class PyTupleIterator;
class PyType;
struct TypePrototype;
struct PyBuffer;

template<typename T> T *as(PyObject *node);
template<typename T> const T *as(const PyObject *node);

}// namespace py
