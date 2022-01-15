#include "LLVMPyUtils.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/Value.hpp"
#include "runtime/types/builtin.hpp"
#include "vm/VM.hpp"

using namespace py;

template<> int64_t from_args(PyTuple *args, size_t index)
{
	ASSERT(index < args->elements().size());
	const auto &el = args->elements()[index];
	ASSERT(std::holds_alternative<Number>(el) || std::holds_alternative<PyObject *>(el))
	if (std::holds_alternative<Number>(el)) {
		auto n = std::get<Number>(el);
		if (std::holds_alternative<int64_t>(n.value)) {
			return std::get<int64_t>(n.value);
		} else {
			TODO();
		}
	}
	auto *obj = std::get<PyObject *>(el);
	ASSERT(obj->type() == integer())
	return as<PyInteger>(obj)->as_i64();
}

template<> PyObject *from(int64_t value) { return PyObject::from(Number{ value }); }

PyObject *create_native_function(const std::string &name,
	std::function<PyObject *(PyTuple *, PyDict *)> &&func)
{
	return PyNativeFunction::create(name, func);
}

int64_t from_args_i64(int8_t *args, int64_t idx)
{
	return from_args<int64_t>(bit_cast<PyTuple *>(args), idx);
}

int8_t *from_i64(int64_t value) { return bit_cast<int8_t *>(from<int64_t>(value)); }