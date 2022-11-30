#include "DictUpdate.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyTuple.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> DictUpdate::execute(VirtualMachine &vm, Interpreter &) const
{
	auto &dst = vm.reg(m_dst);
	auto &src = vm.reg(m_src);

	ASSERT(std::holds_alternative<PyObject *>(dst));
	ASSERT(std::holds_alternative<PyObject *>(src));

	auto *dst_dict = as<PyDict>(std::get<PyObject *>(dst));
	ASSERT(dst_dict);
	auto args = PyTuple::create(std::get<PyObject *>(src), py_true());
	return dst_dict->merge(args.unwrap(), nullptr);
}

std::vector<uint8_t> DictUpdate::serialize() const
{
	std::vector<uint8_t> bytes{
		DICT_UPDATE,
		m_dst,
		m_src,
	};
	return bytes;
}
