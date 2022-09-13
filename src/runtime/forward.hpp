#pragma once

namespace py {

class PyAsyncGenerator;
class PyCell;
class PyCode;
class PyCoroutine;
class PyDict;
class PyFloat;
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
class PySlice;
class PyString;
class PyTraceback;
class PyTuple;
class PyType;
struct TypePrototype;

template<typename T> T *as(PyObject *node);
template<typename T> const T *as(const PyObject *node);

}// namespace py