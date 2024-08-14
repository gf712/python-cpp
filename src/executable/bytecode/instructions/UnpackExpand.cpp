#include "UnpackExpand.hpp"

#include "runtime/PyInteger.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyNumber.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/ValueError.hpp"
#include "vm/VM.hpp"

#include "../serialization/serialize.hpp"

using namespace py;

PyResult<Value> UnpackExpand::execute(VirtualMachine &vm, Interpreter &) const
{
	const auto &source = vm.reg(m_source);

	return [&]() -> PyResult<Value> {
		if (auto *obj = std::get_if<PyObject *>(&source)) {
			if (auto *pytuple = as<PyTuple>(*obj)) {
				if (pytuple->elements().size() < m_destination.size()) {
					return Err(
						value_error("not enough values to unpack (expected at least {}, got 0)",
							m_destination.size()));
				} else {
					size_t i = 0;
					for (; i < m_destination.size(); ++i) {
						vm.reg(m_destination[i]) = pytuple->elements()[i];
					}
					std::vector<Value> rest;
					for (; i < pytuple->elements().size(); ++i) {
						rest.push_back(pytuple->elements()[i]);
					}
					return PyList::create(std::move(rest)).and_then([this, &vm](auto *rest) {
						vm.reg(m_rest) = rest;
						return Ok(Value{ py_none() });
					});
				}
			} else if (auto *pylist = as<PyList>(*obj)) {
				if (pylist->elements().size() < m_destination.size()) {
					return Err(value_error("not enough values to unpack (expected {}, got {})",
						m_destination.size(),
						pylist->elements().size()));
				} else {
					size_t i = 0;
					for (; i < m_destination.size(); ++i) {
						vm.reg(m_destination[i]) = pylist->elements()[i];
					}
					std::vector<Value> rest;
					for (; i < pylist->elements().size(); ++i) {
						rest.push_back(pylist->elements()[i]);
					}
					return PyList::create(std::move(rest)).and_then([this, &vm](auto *rest) {
						vm.reg(m_rest) = rest;
						return Ok(Value{ py_none() });
					});
				}
			} else {
				const auto mapping = (*obj)->as_mapping();
				if (mapping.is_err()) { return Err(mapping.unwrap_err()); }
				const auto source_size = [&] {
					[[maybe_unused]] RAIIStoreNonCallInstructionData non_call_instruction_data;
					return mapping.unwrap().len();
				}();
				if (source_size.is_err()) { return Err(source_size.unwrap_err()); }
				auto len = source_size.unwrap();
				if (len < m_destination.size()) {
					return Err(value_error("not enough values to unpack (expected {}, got {})",
						m_destination.size(),
						len));
				}
				TODO();
			}
		} else if (std::holds_alternative<Number>(source)) {
			return Err(type_error("cannot unpack non-iterable int object"));
		} else if (std::holds_alternative<String>(source)) {
			const auto str = std::get<String>(source);
			if (str.s.size() < m_destination.size()) {
				return Err(value_error("not enough values to unpack (expected {}, got {})",
					m_destination.size(),
					str.s.size()));
			} else {
				size_t i = 0;
				for (; i < m_destination.size(); ++i) {
					vm.reg(m_destination[i]) = String{ std::string{ str.s[i] } };
				}
				std::vector<Value> rest;
				for (; i < str.s.size(); ++i) { rest.push_back(String{ std::string{ str.s[i] } }); }
				return PyList::create(std::move(rest)).and_then([this, &vm](auto *rest) {
					vm.reg(m_rest) = rest;
					return Ok(Value{ py_none() });
				});
			}
		} else if (std::holds_alternative<Bytes>(source)) {
			const auto bytes = std::get<Bytes>(source);
			if (bytes.b.size() < m_destination.size()) {
				return Err(value_error("not enough values to unpack (expected {}, got {})",
					m_destination.size(),
					bytes.b.size()));
			} else {
				size_t i = 0;
				for (; i < m_destination.size(); ++i) {
					vm.reg(m_destination[i]) = Number{ std::to_integer<int64_t>(bytes.b[i]) };
				}
				std::vector<Value> rest;
				for (; i < bytes.b.size(); ++i) {
					rest.push_back(Number{ std::to_integer<int64_t>(bytes.b[i]) });
				}
				return PyList::create(std::move(rest)).and_then([this, &vm](auto *rest) {
					vm.reg(m_rest) = rest;
					return Ok(Value{ py_none() });
				});
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

std::vector<uint8_t> UnpackExpand::serialize() const
{
	std::vector<uint8_t> bytes{
		UNPACK_SEQUENCE,
	};
	::serialize(m_destination, bytes);
	::serialize(m_rest, bytes);
	::serialize(m_source, bytes);

	return bytes;
}
