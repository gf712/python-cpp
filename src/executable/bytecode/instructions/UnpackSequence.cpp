#include "UnpackSequence.hpp"

#include "runtime/PyInteger.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyNumber.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/ValueError.hpp"

using namespace py;

void UnpackSequence::execute(VirtualMachine &vm, Interpreter &) const
{
	const auto &source = vm.reg(m_source);

	if (auto *obj = std::get_if<PyObject *>(&source)) {
		if (auto *pytuple = as<PyTuple>(*obj)) {
			if (pytuple->elements().size() > m_destination.size()) {
				value_error("too many values to unpack (expected {})", m_destination.size());
			} else if (pytuple->elements().size() < m_destination.size()) {
				value_error("not enought values to unpack (expected {}, got {})",
					m_destination.size(),
					pytuple->elements().size());
			} else {
				size_t idx{ 0 };
				for (const auto &el : pytuple->elements()) { vm.reg(m_destination[idx++]) = el; }
			}
		} else if (auto *pylist = as<PyList>(*obj)) {
			if (pylist->elements().size() > m_destination.size()) {
				value_error("too many values to unpack (expected {})", m_destination.size());
			} else if (pylist->elements().size() < m_destination.size()) {
				value_error("not enought values to unpack (expected {}, got {})",
					m_destination.size(),
					pylist->elements().size());
			} else {
				size_t idx{ 0 };
				for (const auto &el : pylist->elements()) { vm.reg(m_destination[idx++]) = el; }
			}
		} else {
			const auto *source_size = (*obj)->len();
			if (auto *pynum = as<PyInteger>(source_size)) {
				if (pynum->value() != Number{ static_cast<int64_t>(m_destination.size()) }) {
					value_error("too many values to unpack (expected {})", m_destination.size());
					return;
				}
				TODO();
			}
			TODO();
		}
	} else if (std::holds_alternative<Number>(source)) {
		type_error("cannot unpack non-iterable int object");
	} else if (std::holds_alternative<String>(source)) {
		const auto str = std::get<String>(source);
		if (str.s.size() > m_destination.size()) {
			value_error("too many values to unpack (expected {})", m_destination.size());
		} else if (str.s.size() < m_destination.size()) {
			value_error("not enought values to unpack (expected {}, got {})",
				m_destination.size(),
				str.s.size());
		} else {
			size_t idx{ 0 };
			for (const auto &el : str.s) {
				vm.reg(m_destination[idx++]) = String{ std::string{ el } };
			}
		}
	} else if (std::holds_alternative<Bytes>(source)) {
		const auto bytes = std::get<Bytes>(source);
		if (bytes.b.size() > m_destination.size()) {
			value_error("too many values to unpack (expected {})", m_destination.size());
		} else if (bytes.b.size() < m_destination.size()) {
			value_error("not enought values to unpack (expected {}, got {})",
				m_destination.size(),
				bytes.b.size());
		} else {
			size_t idx{ 0 };
			for (const auto &el : bytes.b) {
				vm.reg(m_destination[idx++]) = Number{ std::to_integer<int64_t>(el) };
			}
		}
	} else if (std::holds_alternative<Ellipsis>(source)) {
		type_error("cannot unpack non-iterable ellipsis object");
	} else {
		const auto val = std::get<NameConstant>(source).value;
		if (std::holds_alternative<bool>(val)) {
			type_error("cannot unpack non-iterable bool object");
		} else {
			type_error("cannot unpack non-iterable NoneType object");
		}
	}
}

std::vector<uint8_t> UnpackSequence::serialize() const
{
	TODO();
	return {
		UNPACK_SEQUENCE,
		m_source,
	};
}
