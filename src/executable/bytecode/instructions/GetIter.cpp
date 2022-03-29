#include "GetIter.hpp"

using namespace py;

void GetIter::execute(VirtualMachine &vm, Interpreter &) const
{
	auto iterable_value = vm.reg(m_src);
	if (auto *iterable_object = std::get_if<PyObject *>(&iterable_value)) {
		vm.reg(m_dst) = (*iterable_object)->iter();
	} else {
		vm.reg(m_dst) = std::visit(
			[](const auto &value) { return PyObject::from(value)->iter(); }, iterable_value);
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