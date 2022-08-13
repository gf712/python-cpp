#pragma once

namespace py {

class PyCell;
class PyCode;
class PyDict;
class PyFloat;
class PyFunction;
class PyFrame;
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