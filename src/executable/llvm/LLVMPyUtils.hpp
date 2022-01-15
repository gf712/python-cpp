#pragma once

#include <functional>
#include <string>

namespace py {
class PyDict;
class PyObject;
class PyTuple;
}// namespace py

template<typename T> py::PyObject *from(T);
template<typename T> T from_args(py::PyTuple *args, size_t idx);

py::PyObject *create_native_function(const std::string &name,
	std::function<py::PyObject *(py::PyTuple *, py::PyDict *)> &&);

extern "C" {
int64_t from_args_i64(int8_t *args, int64_t idx);
int8_t *from_i64(int64_t value);
}