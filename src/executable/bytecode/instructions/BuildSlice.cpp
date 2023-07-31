#include "BuildSlice.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PySlice.hpp"
#include "vm/VM.hpp"

using namespace py;

PyResult<Value> BuildSlice::execute(VirtualMachine &vm, Interpreter &) const
{
	if (m_step) {
		auto start = vm.reg(*m_start);
		auto end = vm.reg(*m_end);
		auto step = vm.reg(*m_step);

		auto start_obj = PyObject::from(start);
		if (start_obj.is_err()) return start_obj;

		auto end_obj = PyObject::from(end);
		if (end_obj.is_err()) return end_obj;

		auto step_obj = PyObject::from(step);
		if (step_obj.is_err()) return step_obj;

		return PySlice::create(start_obj.unwrap(), end_obj.unwrap(), step_obj.unwrap())
			.and_then([&vm, this](PySlice *slice) {
				vm.reg(m_dst) = slice;
				return Ok(slice);
			});
	} else {
		auto start = vm.reg(*m_start);
		auto end = vm.reg(*m_end);

		auto start_obj = PyObject::from(start);
		if (start_obj.is_err()) return start_obj;

		auto end_obj = PyObject::from(end);
		if (end_obj.is_err()) return end_obj;

		return PySlice::create(start_obj.unwrap(), end_obj.unwrap(), py_none())
			.and_then([&vm, this](PySlice *slice) {
				vm.reg(m_dst) = slice;
				return Ok(slice);
			});
	}
}

std::vector<uint8_t> BuildSlice::serialize() const
{
	if (m_step) {
		ASSERT(m_start.has_value());
		ASSERT(m_end.has_value());
		ASSERT(m_step.has_value());
		return {
			BUILD_SLICE,
			m_dst,
			uint8_t{ 3 },
			static_cast<uint8_t>(*m_start),
			static_cast<uint8_t>(*m_end),
			static_cast<uint8_t>(*m_step),
		};
	} else {
		ASSERT(m_start.has_value());
		ASSERT(m_end.has_value());
		return {
			BUILD_SLICE,
			m_dst,
			uint8_t{ 2 },
			static_cast<uint8_t>(*m_start),
			static_cast<uint8_t>(*m_end),
		};
	}
}