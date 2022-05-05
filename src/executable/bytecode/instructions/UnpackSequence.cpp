#include "UnpackSequence.hpp"

#include "runtime/PyInteger.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyNumber.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/ValueError.hpp"

using namespace py;

PyResult<Value> UnpackSequence::execute(VirtualMachine &vm, Interpreter &) const
{
	const auto &source = vm.reg(m_source);

	return [&]() -> PyResult<Value> {
		if (auto *obj = std::get_if<PyObject *>(&source)) {
			if (auto *pytuple = as<PyTuple>(*obj)) {
				if (pytuple->elements().size() > m_destination.size()) {
					return Err(value_error(
						"too many values to unpack (expected {})", m_destination.size()));
				} else if (pytuple->elements().size() < m_destination.size()) {
					return Err(value_error("not enought values to unpack (expected {}, got {})",
						m_destination.size(),
						pytuple->elements().size()));
				} else {
					size_t idx{ 0 };
					for (const auto &el : pytuple->elements()) {
						vm.reg(m_destination[idx++]) = el;
					}
					return Ok(Value{ py_none() });
				}
			} else if (auto *pylist = as<PyList>(*obj)) {
				if (pylist->elements().size() > m_destination.size()) {
					return Err(value_error(
						"too many values to unpack (expected {})", m_destination.size()));
				} else if (pylist->elements().size() < m_destination.size()) {
					return Err(value_error("not enought values to unpack (expected {}, got {})",
						m_destination.size(),
						pylist->elements().size()));
				} else {
					size_t idx{ 0 };
					for (const auto &el : pylist->elements()) { vm.reg(m_destination[idx++]) = el; }
					return Ok(Value{ py_none() });
				}
			} else {
				const auto source_size = (*obj)->len();
				if (source_size.is_err()) { return Err(source_size.unwrap_err()); }
				auto len = source_size.unwrap();
				if (len != m_destination.size()) {
					return Err(value_error(
						"too many values to unpack (expected {})", m_destination.size()));
				}
				TODO();
			}
		} else if (std::holds_alternative<Number>(source)) {
			return Err(type_error("cannot unpack non-iterable int object"));
		} else if (std::holds_alternative<String>(source)) {
			const auto str = std::get<String>(source);
			if (str.s.size() > m_destination.size()) {
				return Err(
					value_error("too many values to unpack (expected {})", m_destination.size()));
			} else if (str.s.size() < m_destination.size()) {
				return Err(value_error("not enought values to unpack (expected {}, got {})",
					m_destination.size(),
					str.s.size()));
			} else {
				size_t idx{ 0 };
				for (const auto &el : str.s) {
					vm.reg(m_destination[idx++]) = String{ std::string{ el } };
				}
				return Ok(Value{ py_none() });
			}
		} else if (std::holds_alternative<Bytes>(source)) {
			const auto bytes = std::get<Bytes>(source);
			if (bytes.b.size() > m_destination.size()) {
				return Err(
					value_error("too many values to unpack (expected {})", m_destination.size()));
			} else if (bytes.b.size() < m_destination.size()) {
				return Err(value_error("not enought values to unpack (expected {}, got {})",
					m_destination.size(),
					bytes.b.size()));
			} else {
				size_t idx{ 0 };
				for (const auto &el : bytes.b) {
					vm.reg(m_destination[idx++]) = Number{ std::to_integer<int64_t>(el) };
				}
				return Ok(Value{ py_none() });
			}
		} else if (std::holds_alternative<Ellipsis>(source)) {
			return Err(type_error("cannot unpack non-iterable ellipsis object"));
		} else {
			const auto val = std::get<NameConstant>(source).value;
			if (std::holds_alternative<bool>(val)) {
				return Err(type_error("cannot unpack non-iterable bool object"));
			} else {
				return Err(type_error("cannot unpack non-iterable NoneType object"));
			}
		}
	}();
}

std::vector<uint8_t> UnpackSequence::serialize() const
{
	TODO();
	return {
		UNPACK_SEQUENCE,
		m_source,
	};
}
