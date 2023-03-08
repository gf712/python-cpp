#include "GetIter.hpp"
#include "runtime/PyObject.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> GetIter::execute(VirtualMachine &vm, Interpreter &) const
{
	auto iterable_value = vm.reg(m_src);
	auto result = [&]() {
		if (auto *iterable_object = std::get_if<PyObject *>(&iterable_value)) {
			return (*iterable_object)->iter();
		} else {
			return std::visit(
				[](const auto &value) {
					if (auto obj = PyObject::from(value); obj.is_ok()) {
						return obj.unwrap()->iter();
					} else {
						return obj;
					}
				},
				iterable_value);
		}
	}();
	if (result.is_ok()) {
		vm.reg(m_dst) = result.unwrap();
		return Ok(Value{ result.unwrap() });
	} else {
		return Err(result.unwrap_err());
	}
}

std::vector<uint8_t> GetIter::serialize() const
{
	return {
		GET_ITER,
		m_dst,
		m_src,
	};
}
