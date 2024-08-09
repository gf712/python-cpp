#include "PyString.hpp"
#include "IndexError.hpp"
#include "KeyError.hpp"
#include "MemoryError.hpp"
#include "NotImplementedError.hpp"
#include "PyBool.hpp"
#include "PyDict.hpp"
#include "PyInteger.hpp"
#include "PyList.hpp"
#include "PyNone.hpp"
#include "PySlice.hpp"
#include "StopIteration.hpp"
#include "SyntaxError.hpp"
#include "TypeError.hpp"
#include "ValueError.hpp"
#include "interpreter/Interpreter.hpp"
#include "runtime/PyBytes.hpp"
#include "runtime/Value.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"
#include "utilities.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <numeric>
#include <ranges>
#include <span>

#include <unicode/stringpiece.h>
#include <unicode/uchar.h>
#include <unicode/umachine.h>
#include <unicode/unistr.h>

namespace py {

template<> PyString *as(PyObject *obj)
{
	if (obj->type() == types::str()) { return static_cast<PyString *>(obj); }
	return nullptr;
}

template<> const PyString *as(const PyObject *obj)
{
	if (obj->type() == types::str()) { return static_cast<const PyString *>(obj); }
	return nullptr;
}

namespace utf8 {
	// Code point <-> UTF-8 conversion
	// First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
	// U+0000	U+007F	0xxxxxxx
	// U+0080	U+07FF	110xxxxx	10xxxxxx
	// U+0800	U+FFFF	1110xxxx	10xxxxxx	10xxxxxx
	// U+10000	[nb 2]U+10FFFF	11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

	size_t codepoint_length(const char c)
	{
		if ((c & 0xf8) == 0xf0)
			return 4;
		else if ((c & 0xf0) == 0xe0)
			return 3;
		else if ((c & 0xe0) == 0xc0)
			return 2;
		else
			return 1;
	}

	size_t codepoint_length(uint32_t codepoint)
	{
		if (codepoint <= 0x00007F) { return 1; }
		if (codepoint <= 0x0007FF) { return 2; }
		if (codepoint <= 0x00FFFF) { return 3; }
		ASSERT(codepoint <= 0x10FFFF);
		return 4;
	}

	// take from http://www.zedwood.com/article/cpp-utf8-char-to-codepoint
	std::optional<uint32_t> codepoint(const char *str, size_t length)
	{
		if (length < 1) return std::nullopt;
		unsigned char u0 = str[0];
		if (u0 <= 127) return u0;
		if (length < 2) return std::nullopt;
		unsigned char u1 = str[1];
		if (u0 >= 192 && u0 <= 223) return (u0 - 192) * 64 + (u1 - 128);
		if (u0 == 0xed && (u1 & 0xa0) == 0xa0) return -1;// code points, 0xd800 to 0xdfff
		if (length < 3) return std::nullopt;
		unsigned char u2 = str[2];
		if (u0 >= 224 && u0 <= 239) return (u0 - 224) * 4096 + (u1 - 128) * 64 + (u2 - 128);
		if (length < 4) return std::nullopt;
		unsigned char u3 = str[3];
		if (u0 >= 240 && u0 <= 247)
			return (u0 - 240) * 262144 + (u1 - 128) * 4096 + (u2 - 128) * 64 + (u3 - 128);
		return std::nullopt;
	}

}// namespace utf8

PyString::PyString(PyType *type) : PyBaseObject(type) {}

PyResult<PyString *> PyString::create(const std::string &value)
{
	auto &heap = VirtualMachine::the().heap();
	auto *result = heap.allocate<PyString>(value);
	if (!result) { return Err(memory_error(sizeof(PyString))); }
	return Ok(result);
}

PyResult<PyString *> PyString::create(PyObject *obj)
{
	if (auto *s = as<PyString>(obj)) {
		return Ok(s);
	} else if (obj->type()->underlying_type().__str__.has_value()) {
		return obj->str();
	} else {
		return obj->repr();
	}
}

PyResult<PyString *> PyString::create(const Bytes &bytes, const std::string &encoding)
{
	if (encoding.empty()) { return PyString::create(bytes.to_string()); }
	if (encoding == "latin1") {
		// based on https://stackoverflow.com/a/4059934
		std::string result;
		for (const auto &byte : bytes.b) {
			if (byte < std::byte{ 128 }) {
				result.push_back(static_cast<char>(byte));
			} else {
				result.push_back(0xc2 + (static_cast<unsigned char>(byte) > 0xbf));
				result.push_back((static_cast<char>(byte) & 0x3f) + 0x80);
			}
		}
		return PyString::create(result);
	} else if (encoding == "utf8") {
		std::string result;
		auto it = bytes.b.begin();
		while (it != bytes.b.end()) {
			if (*it > std::byte{ 127 }) {
				return Err(value_error(
					"'utf-8' codec can't decode byte {} in position {}: invalid start byte",
					*it,
					std::distance(bytes.b.begin(), it)));
			}
			auto length = utf8::codepoint_length(static_cast<char>(*it));
			if (!utf8::codepoint(bit_cast<const char *>(it.base()), length).has_value()) {
				return Err(value_error(
					"'utf-8' codec can't decode byte {} in position {}: invalidutf8 codepoint ",
					*it,
					std::distance(bytes.b.begin(), it)));
			}
			for (size_t i = 0; i < length; ++i) {
				if (it == bytes.b.end()) {
					return Err(value_error(
						"'utf-8' codec can't decode byte {} in position {}: invalidutf8 codepoint ",
						*it,
						std::distance(bytes.b.begin(), it)));
				}
				result.push_back(static_cast<char>(*it));
				it++;
			}
		}
		return PyString::create(result);
	}
	TODO();
}

PyResult<PyObject *> PyString::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	// FIXME: this should use either __str__ or __repr__ rather than relying on first arg being a
	// String
	// FIXME: handle bytes_or_buffer argument
	// FIXME: handle encoding argument
	// FIXME: handle errors argument
	ASSERT(!kwargs || kwargs->map().size() == 0)
	ASSERT(args && args->size() <= 2)
	ASSERT(type == types::str())

	std::string encoding;

	const auto &string = args->elements()[0];
	if (args->size() > 1) {
		auto *el1 = PyObject::from(args->elements()[1]).unwrap();
		if (!el1->type()->issubclass(types::str())) {
			return Err(
				type_error("str() argument 'encoding' must be str, not {}", el1->type()->name()));
		}
		encoding = static_cast<const PyString &>(*el1).value();
	}

	if (std::holds_alternative<String>(string)) {
		return PyString::create(std::get<String>(string).s);
	} else if (std::holds_alternative<PyObject *>(string)) {
		auto s = std::get<PyObject *>(string);
		if (s->type()->issubclass(types::bytes())) {
			return PyString::create(static_cast<const PyBytes &>(*s).value(), encoding);
		} else if (s->type()->issubclass(types::bytearray())) {
			return PyString::create(static_cast<const PyBytes &>(*s).value(), encoding);
		}
		return PyString::create(s);
	} else if (std::holds_alternative<Bytes>(string)) {
		return PyString::create(std::get<Bytes>(string), encoding);
	} else {
		TODO();
	}
}

PyString::PyString(std::string s)
	: PyBaseObject(types::BuiltinTypes::the().str()), m_value(std::move(s))
{}

PyResult<int64_t> PyString::__hash__() const
{
	return Ok(static_cast<int64_t>(std::hash<std::string>{}(m_value)));
}

PyResult<PyObject *> PyString::__repr__() const
{
	return PyString::create(fmt::format("'{}'", m_value));
}

PyResult<PyObject *> PyString::__str__() const { return Ok(const_cast<PyString *>(this)); }

PyResult<PyObject *> PyString::__add__(const PyObject *obj) const
{
	if (auto rhs = as<PyString>(obj)) {
		return PyString::create(m_value + rhs->value());
	} else {
		return Err(type_error("unsupported operand type(s) for +: \'{}\' and \'{}\'",
			type()->name(),
			obj->type()->name()));
	}
}

PyResult<PyObject *> PyString::__mod__(const PyObject *obj) const { return printf(obj); }

PyResult<PyObject *> PyString::__eq__(const PyObject *obj) const
{
	if (this == obj) return Ok(py_true());
	if (auto obj_string = as<PyString>(obj)) {
		return Ok(m_value == obj_string->value() ? py_true() : py_false());
	} else {
		return Err(type_error("'==' not supported between instances of '{}' and '{}'",
			type()->name(),
			obj->type()->name()));
	}
}


PyResult<PyObject *> PyString::__ne__(const PyObject *obj) const
{
	if (this == obj) return Ok(py_false());
	if (auto obj_string = as<PyString>(obj)) {
		return Ok(m_value != obj_string->value() ? py_true() : py_false());
	} else {
		return Err(type_error("'!=' not supported between instances of '{}' and '{}'",
			type()->name(),
			obj->type()->name()));
	}
}

PyResult<PyObject *> PyString::__lt__(const PyObject *obj) const
{
	if (this == obj) return Ok(py_true());
	if (auto obj_string = as<PyString>(obj)) {
		return Ok(m_value < obj_string->value() ? py_true() : py_false());
	} else {
		return Err(type_error("'==' not supported between instances of '{}' and '{}'",
			type()->name(),
			obj->type()->name()));
	}
}

size_t PyString::size() const
{
	size_t size{ 0 };
	for (auto it = m_value.begin(); it != m_value.end();) {
		const auto codepoint_byte_size = utf8::codepoint_length(*it);
		size++;
		it += codepoint_byte_size;
	}
	return size;
}

PyResult<size_t> PyString::__len__() const { return Ok(size()); }

PyResult<bool> PyString::__bool__() const { return Ok(!m_value.empty()); }

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::capitalize() const
{
	auto new_string = m_value;
	new_string[0] = static_cast<char>(std::toupper(new_string[0]));
	return PyString::create(new_string);
}

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::casefold() const
{
	auto new_string = m_value;
	std::transform(new_string.begin(),
		new_string.end(),
		new_string.begin(),
		[](const unsigned char c) -> unsigned char {
			return static_cast<unsigned char>(std::tolower(c));
		});
	return PyString::create(new_string);
}

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::isalnum() const
{
	if (m_value.empty()) { return Ok(py_false()); }

	auto it = std::find_if_not(
		m_value.begin(), m_value.end(), [](const unsigned char c) { return std::isalnum(c); });
	if (it == m_value.end()) {
		return Ok(py_true());
	} else {
		return Ok(py_false());
	}
}

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::isalpha() const
{
	if (m_value.empty()) { return Ok(py_false()); }

	auto it = std::find_if_not(
		m_value.begin(), m_value.end(), [](const unsigned char c) { return std::isalpha(c); });
	if (it == m_value.end()) {
		return Ok(py_true());
	} else {
		return Ok(py_false());
	}
}

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::isdigit() const
{
	if (m_value.empty()) { return Ok(py_false()); }

	auto it = std::find_if_not(
		m_value.begin(), m_value.end(), [](const unsigned char c) { return std::isdigit(c); });
	if (it == m_value.end()) {
		return Ok(py_true());
	} else {
		return Ok(py_false());
	}
}

PyResult<PyObject *> PyString::isascii() const
{
	for (size_t i = 0; i < m_value.size();) {
		const auto length = utf8::codepoint_length(m_value[i]);
		const auto codepoint = utf8::codepoint(m_value.c_str(), length);
		if (codepoint > 0x7F) { return Ok(py_false()); }
		i += length;
	}
	return Ok(py_true());
}

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::islower() const
{
	if (m_value.empty()) { return Ok(py_false()); }

	auto it = std::find_if_not(m_value.begin(), m_value.end(), [](const unsigned char c) {
		return !std::isalpha(c) || std::islower(c);
	});
	if (it == m_value.end()) {
		return Ok(py_true());
	} else {
		return Ok(py_false());
	}
}

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::isupper() const
{
	if (m_value.empty()) { return Ok(py_false()); }

	auto it = std::find_if_not(m_value.begin(), m_value.end(), [](const unsigned char c) {
		return !std::isalpha(c) || std::isupper(c);
	});
	if (it == m_value.end()) {
		return Ok(py_true());
	} else {
		return Ok(py_false());
	}
}

size_t PyString::get_position_from_slice(int64_t pos) const
{
	if (pos < 0) {
		pos = m_value.size() - pos;
		// TODO: handle case where the negative start index is less than size of string
		ASSERT(pos > 0)
	}
	return pos;
}

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::find(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(args && args->size() <= 3 && args->size() > 0)
	ASSERT(!kwargs)

	auto pattern_ = PyObject::from(args->elements()[0]);
	if (pattern_.is_err()) return pattern_;
	PyString *pattern = as<PyString>(pattern_.unwrap());
	PyInteger *start = nullptr;
	PyInteger *end = nullptr;
	size_t result{ std::string::npos };

	if (args->size() >= 2) {
		auto start_ = PyObject::from(args->elements()[1]);
		if (start_.is_err()) return start_;
		start = as<PyInteger>(start_.unwrap());
		// TODO: raise exception when start in not a number
		ASSERT(start)
	}
	if (args->size() == 3) {
		auto end_ = PyObject::from(args->elements()[2]);
		if (end_.is_err()) return end_;
		end = as<PyInteger>(end_.unwrap());
		// TODO: raise exception when end in not a number
		ASSERT(end)
	}

	if (!start && !end) {
		result = m_value.find(pattern->value().c_str());
	} else if (!end) {
		size_t start_ =
			std::visit(overloaded{
						   [this](const auto &val) -> size_t {
							   return get_position_from_slice(static_cast<int64_t>(val));
						   },
						   [this](const mpz_class &val) -> size_t {
							   ASSERT(val.fits_slong_p());
							   return get_position_from_slice(val.get_si());
						   },
					   },
				start->value().value);
		result = m_value.find(pattern->value().c_str(), start_);
	} else {
		size_t start_ =
			std::visit(overloaded{
						   [this](const auto &val) -> size_t {
							   return get_position_from_slice(static_cast<int64_t>(val));
						   },
						   [this](const mpz_class &val) -> size_t {
							   ASSERT(val.fits_slong_p());
							   return get_position_from_slice(val.get_si());
						   },
					   },
				start->value().value);
		size_t end_ = std::visit(overloaded{
									 [this](const auto &val) -> size_t {
										 return get_position_from_slice(static_cast<int64_t>(val));
									 },
									 [this](const mpz_class &val) -> size_t {
										 ASSERT(val.fits_slong_p());
										 return get_position_from_slice(val.get_si());
									 },
								 },
			end->value().value);
		size_t subtring_size = end_ - start_;
		result = m_value.find(pattern->value().c_str(), start_, subtring_size);
	}
	if (result == std::string::npos) {
		return PyInteger::create(int64_t{ -1 });
	} else {
		return PyInteger::create(static_cast<int64_t>(result));
	}
}

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::rfind(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(args && args->size() <= 3 && args->size() > 0)
	ASSERT(!kwargs)

	auto pattern_ = PyObject::from(args->elements()[0]);
	if (pattern_.is_err()) return pattern_;
	PyString *pattern = as<PyString>(pattern_.unwrap());
	PyInteger *start = nullptr;
	PyInteger *end = nullptr;

	if (args->size() >= 2) {
		auto start_ = PyObject::from(args->elements()[1]);
		if (start_.is_err()) return start_;
		start = as<PyInteger>(start_.unwrap());
		// TODO: raise exception when start in not a number
		ASSERT(start)
	}
	if (args->size() == 3) {
		auto end_ = PyObject::from(args->elements()[2]);
		if (end_.is_err()) return end_;
		end = as<PyInteger>(end_.unwrap());
		// TODO: raise exception when end in not a number
		ASSERT(end)
	}

	size_t start_idx = 0;
	size_t end_idx = 0;

	if (!start && !end) {
		start_idx = 0;
		end_idx = m_value.size();
	} else if (!end) {
		start_idx = std::visit(overloaded{
								   [this](const auto &val) -> size_t {
									   return get_position_from_slice(static_cast<int64_t>(val));
								   },
								   [this](const mpz_class &val) -> size_t {
									   ASSERT(val.fits_slong_p());
									   return get_position_from_slice(val.get_si());
								   },
							   },
			start->value().value);
		end_idx = m_value.size();
	} else {
		start_idx = std::visit(overloaded{
								   [this](const auto &val) -> size_t {
									   return get_position_from_slice(static_cast<int64_t>(val));
								   },
								   [this](const mpz_class &val) -> size_t {
									   ASSERT(val.fits_slong_p());
									   return get_position_from_slice(val.get_si());
								   },
							   },
			start->value().value);
		end_idx = std::visit(overloaded{
								 [this](const auto &val) -> size_t {
									 return get_position_from_slice(static_cast<int64_t>(val));
								 },
								 [this](const mpz_class &val) -> size_t {
									 ASSERT(val.fits_slong_p());
									 return get_position_from_slice(val.get_si());
								 },
							 },
			end->value().value);
	}

	std::optional<size_t> result;
	while (start_idx < end_idx) {
		start_idx = m_value.find(pattern->value(), start_idx);
		if (start_idx < end_idx) { result = start_idx++; }
	}

	if (result.has_value()) {
		return PyInteger::create(static_cast<int64_t>(*result));
	} else {
		return PyInteger::create(int64_t{ -1 });
	}
}


// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::count(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(args && args->size() <= 3 && args->size() > 0)
	ASSERT(!kwargs)

	auto pattern_ = PyObject::from(args->elements()[0]);
	if (pattern_.is_err()) return pattern_;
	PyString *pattern = as<PyString>(pattern_.unwrap());
	PyInteger *start = nullptr;
	PyInteger *end = nullptr;
	size_t result{ 0 };

	if (args->size() >= 2) {
		auto start_ = PyObject::from(args->elements()[1]);
		if (start_.is_err()) return start_;
		start = as<PyInteger>(start_.unwrap());
		// TODO: raise exception when start in not a number
		ASSERT(start)
	}
	if (args->size() == 3) {
		auto end_ = PyObject::from(args->elements()[2]);
		if (end_.is_err()) return end_;
		end = as<PyInteger>(end_.unwrap());
		// TODO: raise exception when end in not a number
		ASSERT(end)
	}

	const size_t start_ = [start, this]() {
		if (start) {
			return std::visit(overloaded{
								  [this](const auto &val) -> size_t {
									  return get_position_from_slice(static_cast<int64_t>(val));
								  },
								  [this](const mpz_class &val) -> size_t {
									  ASSERT(val.fits_slong_p());
									  return get_position_from_slice(val.get_si());
								  },
							  },
				start->value().value);
		} else {
			return size_t{ 0 };
		}
	}();

	const size_t end_ = [end, this]() {
		if (end) {
			return std::visit(overloaded{
								  [this](const auto &val) -> size_t {
									  return get_position_from_slice(static_cast<int64_t>(val));
								  },
								  [this](const mpz_class &val) -> size_t {
									  ASSERT(val.fits_slong_p());
									  return get_position_from_slice(val.get_si());
								  },
							  },
				end->value().value);
		} else {
			return m_value.size();
		}
	}();

	auto iter = m_value.begin() + start_;
	const auto end_it = m_value.begin() + end_;

	while (iter != end_it) {
		std::string_view substring{ &(*iter), static_cast<size_t>(std::distance(iter, end_it)) };
		if (auto pos = substring.find(pattern->value()); pos != std::string_view::npos) {
			result++;
			// if we advance to the end of the substring match, do we still have enough of
			// a substring left to find another match
			if (std::distance(iter + pos + pattern->value().size(), end_it)
				>= static_cast<int64_t>(pattern->value().size()))
				iter += pos + pattern->value().size();
			else {
				break;
			}
		} else {
			break;
		}
	}

	return PyInteger::create(static_cast<int64_t>(result));
}

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::startswith(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(args && args->size() <= 3 && args->size() > 0)
	ASSERT(!kwargs)

	auto prefix_ = PyObject::from(args->elements()[0]);
	if (prefix_.is_err()) return prefix_;

	auto prefixes_ = [&prefix_]() -> PyResult<std::vector<std::string>> {
		PyString *prefix = as<PyString>(prefix_.unwrap());
		if (prefix) { return Ok(std::vector<std::string>{ prefix->value() }); }
		PyTuple *prefixes = as<PyTuple>(prefix_.unwrap());
		if (prefixes) {
			std::vector<std::string> els;
			els.reserve(prefixes->size());
			for (const auto &el : prefixes->elements()) {
				auto obj = PyObject::from(el);
				if (obj.is_err()) return Err(obj.unwrap_err());
				PyString *prefix = as<PyString>(obj.unwrap());
				if (!prefix) { return Err(type_error("expected tuple of objects of type str")); }
				els.push_back(prefix->value());
			}
			return Ok(els);
		}
		return Err(type_error("startswith first arg must be str or a tuple of str, not '{}'",
			prefix_.unwrap()->type()->name()));
	}();

	PyInteger *start = nullptr;
	PyInteger *end = nullptr;

	if (args->size() >= 2) {
		auto start_ = PyObject::from(args->elements()[1]);
		if (start_.is_err()) return start_;
		start = as<PyInteger>(start_.unwrap());
		// TODO: raise exception when start in not a number
		ASSERT(start)
	}
	if (args->size() == 3) {
		auto end_ = PyObject::from(args->elements()[2]);
		if (end_.is_err()) return end_;
		end = as<PyInteger>(end_.unwrap());
		// TODO: raise exception when end in not a number
		ASSERT(end)
	}

	if (prefixes_.is_err()) return Err(prefixes_.unwrap_err());

	const auto &prefixes = prefixes_.unwrap();

	const auto result = std::any_of(
		prefixes.begin(), prefixes.end(), [&start, &end, this](const std::string &prefix) {
			if (!start && !end) {
				return m_value.starts_with(prefix);
			} else if (!end) {
				size_t start_ =
					std::visit(overloaded{
								   [this](const auto &val) -> size_t {
									   return get_position_from_slice(static_cast<int64_t>(val));
								   },
								   [this](const mpz_class &val) -> size_t {
									   ASSERT(val.fits_slong_p());
									   return get_position_from_slice(val.get_si());
								   },
							   },
						start->value().value);
				std::string_view substring{ m_value.c_str() + start_, m_value.size() - start_ };
				return substring.starts_with(prefix);
			} else {
				size_t start_ =
					std::visit(overloaded{
								   [this](const auto &val) -> size_t {
									   return get_position_from_slice(static_cast<int64_t>(val));
								   },
								   [this](const mpz_class &val) -> size_t {
									   ASSERT(val.fits_slong_p());
									   return get_position_from_slice(val.get_si());
								   },
							   },
						start->value().value);
				size_t end_ =
					std::visit(overloaded{
								   [this](const auto &val) -> size_t {
									   return get_position_from_slice(static_cast<int64_t>(val));
								   },
								   [this](const mpz_class &val) -> size_t {
									   ASSERT(val.fits_slong_p());
									   return get_position_from_slice(val.get_si());
								   },
							   },
						end->value().value);
				std::string_view substring{ m_value.c_str() + start_, end_ - start_ };
				return substring.starts_with(prefix);
			}
		});

	return Ok(result ? py_true() : py_false());
}

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::endswith(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(args && args->size() <= 3 && args->size() > 0)
	ASSERT(!kwargs)

	PyInteger *start = nullptr;
	PyInteger *end = nullptr;

	if (args->size() >= 2) {
		auto start_ = PyObject::from(args->elements()[1]);
		if (start_.is_err()) return start_;
		start = as<PyInteger>(start_.unwrap());
		// TODO: raise exception when start in not a number
		ASSERT(start)
	}
	if (args->size() == 3) {
		auto end_ = PyObject::from(args->elements()[2]);
		if (end_.is_err()) return end_;
		end = as<PyInteger>(end_.unwrap());
		// TODO: raise exception when end in not a number
		ASSERT(end)
	}

	std::optional<size_t> start_;
	std::optional<size_t> end_;
	if (start) {
		start_ = std::visit(overloaded{
								[this](const auto &val) -> size_t {
									return get_position_from_slice(static_cast<int64_t>(val));
								},
								[this](const mpz_class &val) -> size_t {
									ASSERT(val.fits_slong_p());
									return get_position_from_slice(val.get_si());
								},
							},
			start->value().value);
	}

	if (end) {
		end_ = std::visit(overloaded{
							  [this](const auto &val) -> size_t {
								  return get_position_from_slice(static_cast<int64_t>(val));
							  },
							  [this](const mpz_class &val) -> size_t {
								  ASSERT(val.fits_slong_p());
								  return get_position_from_slice(val.get_si());
							  },
						  },
			end->value().value);
	}

	auto endswith_impl = [this, start = start_, end = end_](std::string_view suffix) {
		if (!start.has_value() && !end.has_value()) {
			return m_value.ends_with(suffix);
		} else if (!end.has_value()) {
			std::string_view substring{ m_value.c_str() + *start, m_value.size() - *start };
			return substring.ends_with(suffix);
		} else {
			std::string_view substring{ m_value.c_str() + *start, *end - *start };
			return substring.ends_with(suffix);
		}
	};

	auto suffix_ = PyObject::from(args->elements()[0]);
	if (suffix_.is_err()) return suffix_;
	if (auto *suffix = as<PyString>(suffix_.unwrap())) {
		return Ok(endswith_impl(suffix->value()) ? py_true() : py_false());
	} else if (auto *suffix_tuple = as<PyTuple>(suffix_.unwrap())) {
		for (const auto &el : suffix_tuple->elements()) {
			auto obj = PyObject::from(el);
			if (obj.is_err()) { return obj; }
			if (auto *suffix = as<PyString>(obj.unwrap())) {
				if (endswith_impl(suffix->value())) { return Ok(py_true()); }
			} else {
				return Err(type_error("tuple for endswith must only contain str, not '{}'",
					obj.unwrap()->type()->name()));
			}
		}
	} else {
		return Err(type_error("endswith first arg must be str or a tuple of str, not '{}'",
			suffix_.unwrap()->type()->name()));
	}

	return Ok(py_false());
}

PyResult<PyObject *> PyString::join(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(args && args->size() == 1)
	ASSERT(!kwargs)

	auto iterable = PyObject::from(args->elements()[0]);
	if (iterable.is_err()) return iterable;

	auto iterator_ = iterable.unwrap()->iter();
	if (iterator_.is_err()) return iterator_;

	auto *iterator = iterator_.unwrap();
	auto value = iterator->next();
	size_t idx = 0;
	std::string new_string;
	while (value.is_ok()) {
		if (!as<PyString>(value.unwrap())) {
			return Err(type_error("sequence item {}: expected str instance, {} found",
				idx,
				value.unwrap()->type()->name()));
		}
		new_string.append(as<PyString>(value.unwrap())->value());
		value = iterator->next();
		if (value.is_ok()) { new_string.append(m_value); }
	};

	if (value.is_err() && value.unwrap_err()->type() != stop_iteration()->type()) { return value; }
	return PyString::create(new_string);
}

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::lower() const
{
	auto new_string = m_value;
	std::transform(new_string.begin(),
		new_string.end(),
		new_string.begin(),
		[](const unsigned char c) -> unsigned char {
			return static_cast<unsigned char>(std::tolower(c));
		});
	return PyString::create(new_string);
}

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::upper() const
{
	auto new_string = m_value;
	std::transform(new_string.begin(),
		new_string.end(),
		new_string.begin(),
		[](const unsigned char c) -> unsigned char {
			return static_cast<unsigned char>(std::toupper(c));
		});
	return PyString::create(new_string);
}

PyResult<PyObject *> PyString::rpartition(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(args && args->size() == 1)
	ASSERT(!kwargs || kwargs->size() == 0)

	auto sep_ = PyObject::from(args->elements()[0]);
	if (sep_.is_err()) return sep_;
	auto *sep_obj = as<PyString>(sep_.unwrap());
	ASSERT(sep_obj)

	const auto &sep = sep_obj->value();

	auto split_index = m_value.rfind(sep);

	if (split_index == std::string::npos) {
		return PyTuple::create(PyString::create("").unwrap(),
			PyString::create("").unwrap(),
			const_cast<PyString *>(this));
	}

	auto lhs = m_value.substr(0, split_index);
	std::string rhs{ m_value.begin() + split_index + sep.size(), m_value.end() };
	return PyTuple::create(PyString::create(lhs).unwrap(), sep_obj, PyString::create(rhs).unwrap());
}

PyResult<PyObject *> PyString::strip(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(!kwargs || kwargs->size() == 0)

	const auto chars = [args]() -> PyResult<std::vector<uint32_t>> {
		if (!args || args->size() == 0) { return Ok(std::vector<uint32_t>{}); }
		auto args0 = PyObject::from(args->elements()[0]);

		auto str = as<PyString>(args0.unwrap());
		if (!str) { return Err(type_error("")); }
		return Ok(str->codepoints());
	}();

	if (chars.is_err()) return Err(chars.unwrap_err());

	if (chars.unwrap().empty()) {
		const auto it_start = std::find_if(
			m_value.begin(), m_value.end(), [](const auto &el) { return !std::isspace(el); });
		const auto it_end = std::find_if(
			m_value.rbegin(), m_value.rend(), [](const auto &el) { return !std::isspace(el); });

		if (it_start == m_value.begin() && it_end == m_value.rend()) {
			return Ok(const_cast<PyString *>(this));
		}
		std::string result{ it_start, it_end.base() };
		return PyString::create(result);
	} else {
		const auto codepoints = this->codepoints();
		const auto patterns = chars.unwrap();

		auto contains = [patterns](const int32_t &cp) {
			return std::find(patterns.begin(), patterns.end(), cp) != patterns.end();
		};

		auto codepoints_start_it = codepoints.begin();
		size_t string_index_start = 0;
		while (contains(*codepoints_start_it)) {
			string_index_start += utf8::codepoint_length(*codepoints_start_it);
			codepoints_start_it++;
		}

		auto codepoints_end_it = codepoints.rbegin();
		size_t string_index_end = m_value.size();
		while (contains(*codepoints_end_it)) {
			string_index_end -= utf8::codepoint_length(*codepoints_end_it);
			codepoints_end_it++;
		}

		if (string_index_start == 0 && string_index_end == m_value.size()) {
			return Ok(const_cast<PyString *>(this));
		} else {
			ASSERT(string_index_end >= string_index_start);
			const auto &result =
				m_value.substr(string_index_start, string_index_end - string_index_start);
			return PyString::create(result);
		}
	}
}

PyResult<PyObject *> PyString::rstrip(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(!kwargs || kwargs->size() == 0)

	const auto chars = [args]() -> PyResult<std::vector<uint32_t>> {
		if (!args || args->size() == 0) { return Ok(std::vector<uint32_t>{}); }
		auto args0 = PyObject::from(args->elements()[0]);

		auto str = as<PyString>(args0.unwrap());
		if (!str) { return Err(type_error("")); }
		return Ok(str->codepoints());
	}();

	if (chars.is_err()) return Err(chars.unwrap_err());

	if (chars.unwrap().empty()) {
		const auto it = std::find_if(
			m_value.rbegin(), m_value.rend(), [](const auto &el) { return !std::isspace(el); });
		if (it != m_value.rend()) {
			const size_t idx = std::distance(it, m_value.rend());
			std::string result{ m_value.begin(), m_value.begin() + idx };
			return PyString::create(result);
		} else {
			return Ok(const_cast<PyString *>(this));
		}
	} else {
		const auto codepoints = this->codepoints();
		auto codepoints_it = codepoints.rbegin();
		size_t string_index = 0;
		const auto patterns = chars.unwrap();

		auto contains = [patterns](const int32_t &cp) {
			return std::find(patterns.begin(), patterns.end(), cp) != patterns.end();
		};

		while (contains(*codepoints_it)) {
			string_index += utf8::codepoint_length(*codepoints_it);
			codepoints_it++;
		}

		if (string_index == 0) {
			return Ok(const_cast<PyString *>(this));
		} else {
			const auto &result = m_value.substr(0, m_value.size() - string_index);
			return PyString::create(result);
		}
	}
}

PyResult<PyList *> PyString::split(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(!kwargs || kwargs->map().empty());

	const auto sep_ = [args]() -> PyResult<std::vector<uint32_t>> {
		if (!args || args->size() == 0) { return Ok(std::vector<uint32_t>{}); }
		auto args0 = PyObject::from(args->elements()[0]);

		if (args0.unwrap() == py_none()) { return Ok(std::vector<uint32_t>{}); }
		auto str = as<PyString>(args0.unwrap());
		if (!str) { return Err(type_error("")); }
		if (str->value().empty()) { return Err(value_error("empty separator")); }
		return Ok(str->codepoints());
	}();

	if (sep_.is_err()) { return Err(sep_.unwrap_err()); }
	const auto &sep = sep_.unwrap();
	if (m_value.empty() && !sep.empty()) {
		// Splitting an empty string with a specified separator returns ['']
		return PyList::create(std::vector<Value>{ PyString::create("").unwrap() });
	}

	const auto maxsplit_ = [this, args]() -> PyResult<BigIntType> {
		if (!args || args->size() < 2) { return Ok(BigIntType{ m_value.size() }); }
		auto args1 = PyObject::from(args->elements()[1]);

		auto maxsplit = as<PyInteger>(args1.unwrap());
		if (!maxsplit) { return Err(type_error("")); }
		if (maxsplit->as_big_int() == -1) { return Ok(BigIntType{ m_value.size() }); }
		return Ok(maxsplit->as_big_int());
	}();

	if (maxsplit_.is_err()) { return Err(maxsplit_.unwrap_err()); }
	const auto &maxsplit = maxsplit_.unwrap();

	auto result_ = PyList::create();
	if (result_.is_err()) { return result_; }
	auto *result = result_.unwrap();

	if (!sep.empty()) {
		size_t start = 0;
		const auto cps = codepoints();
		for (size_t i = 0; i < cps.size();) {
			if (result->elements().size() >= maxsplit) { break; }
			bool is_match = true;
			for (size_t j = 0; j < sep.size(); ++j) {
				if (cps[i + j] != sep[j]) {
					is_match = false;
					break;
				}
			}
			if (is_match) {
				auto el = PyString::create(m_value.substr(start, i - start));
				if (el.is_err()) { return Err(el.unwrap_err()); }
				result->elements().push_back(el.unwrap());
				i += sep.size();
				start = i;
			} else {
				++i;
			}
		}
		// handle remainder
		auto el = PyString::create(m_value.substr(start, m_value.size() - start));
		if (el.is_err()) { return Err(el.unwrap_err()); }
		result->elements().push_back(el.unwrap());
	} else {
		// If sep is not specified or is None, a different splitting algorithm is applied: runs of
		// consecutive whitespace are regarded as a single separator, and the result will contain no
		// empty strings at the start or end if the string has leading or trailing whitespace.
		// Consequently, splitting an empty string or a string consisting of just whitespace with a
		// None separator returns [].
		size_t start = 0;
		size_t end = start;
		for (const auto &el : m_value) {
			if (result->elements().size() >= maxsplit) { break; }
			if (std::isspace(el)) {
				if (end != start) {
					if (!std::isspace(m_value[start])) {
						auto el = PyString::create(m_value.substr(start, end - start));
						if (el.is_err()) { return Err(el.unwrap_err()); }
						result->elements().push_back(el.unwrap());
					}
					end++;
					start = end;
				} else {
					start++;
					end++;
				}
			} else {
				end++;
			}
		}
		// handle remainder
		std::string_view remainder{ m_value.begin() + start, m_value.end() };
		if (!remainder.empty()) {
			auto el = PyString::create(std::string{ remainder });
			if (el.is_err()) { return Err(el.unwrap_err()); }
			result->elements().push_back(el.unwrap());
		}
	}

	return Ok(result);
}


std::vector<uint32_t> PyString::codepoints() const
{
	std::vector<uint32_t> codepoints;

	for (size_t i = 0; i < m_value.size();) {
		const auto length = utf8::codepoint_length(m_value[i]);
		const auto cp = utf8::codepoint(m_value.c_str() + i, length);
		ASSERT(cp.has_value());
		codepoints.push_back(*cp);
		i += length;
	}
	return codepoints;
}


std::optional<uint32_t> PyString::codepoint() const
{
	if (auto codepoint_length = utf8::codepoint_length(m_value[0]);
		codepoint_length != m_value.size()) {
		return {};
	} else {
		return utf8::codepoint(m_value.c_str(), codepoint_length);
	}
}

PyType *PyString::static_type() const { return types::str(); }

PyResult<PyObject *> PyString::__iter__() const
{
	auto &heap = VirtualMachine::the().heap();
	auto *it = heap.allocate<PyStringIterator>(*this);
	if (!it) { return Err(memory_error(sizeof(PyStringIterator))); }
	return Ok(it);
}

PyResult<PyObject *> PyString::__getitem__(PyObject *index)
{
	if (auto index_int = as<PyInteger>(index)) {
		const auto i = index_int->as_i64();
		return (*this)[i];
	} else if (auto slice = as<PySlice>(index)) {
		const auto codepoints = this->codepoints();
		auto indices_ = slice->unpack();
		if (indices_.is_err()) return Err(indices_.unwrap_err());
		const auto [start_, end_, step] = indices_.unwrap();

		const auto [start, end, slice_length] =
			PySlice::adjust_indices(start_, end_, step, codepoints.size());

		if (slice_length == 0) { return PyList::create(); }
		if (start == 0 && end == static_cast<int64_t>(codepoints.size()) && step == 1) {
			return Ok(this);
		}

		std::string new_str;
		std::string str;

		for (int64_t idx = start, i = 0; i < slice_length; idx += step, ++i) {
			const auto cp = codepoints[idx];
			// is this a valid unicode codepoint?
			ASSERT(cp < 0x110000);
			icu::UnicodeString uni_str(static_cast<UChar32>(cp));
			str.clear();
			uni_str.toUTF8String(str);
			new_str += str;
		}
		return PyString::create(new_str);
	} else {
		return Err(
			type_error("str indices must be integers or slices, not {}", index->type()->name()));
	}
}

PyResult<PyObject *> PyString::operator[](int64_t index) const
{
	const int64_t str_size = size();
	if (index < 0) {
		if (std::abs(index) > str_size) { return Err(index_error("string index out of range")); }
		index += str_size;
	}

	ASSERT(index >= 0);

	if (index >= str_size) { return Err(index_error("string index out of range")); }

	const auto cp = codepoints()[index];
	ASSERT(cp < 0x110000);
	icu::UnicodeString uni_str(static_cast<UChar32>(cp));
	std::string str;
	uni_str.toUTF8String(str);
	return PyString::create(str);
}

PyResult<std::string> PyString::FormatSpec::apply(PyObject *obj) const
{
	if (mapping.has_value()) { TODO(); }
	if (conversion_flag.has_value()) { TODO(); }
	if (minimum_width.has_value()) { TODO(); }
	if (precision.has_value()) { TODO(); }
	if (mapping.has_value()) { TODO(); }
	ASSERT(conversion_type.has_value());
	switch (*conversion_type) {
	case 's': {
		return PyString::create(obj).and_then([](PyString *str) { return Ok(str->value()); });
	} break;
	case 'r': {
		return obj->repr().and_then([](PyString *str) { return Ok(str->value()); });
	} break;
	default:
		return Err(
			not_implemented_error("printf conversion type '{}' not implemented", *conversion_type));
	}
}

PyResult<PyString *> PyString::printf(const PyObject *values) const
{
	static constexpr std::array conversion_types = {
		'd', 'i', 'o', 'u', 'x', 'X', 'e', 'E', 'f', 'F', 'g', 'G', 'c', 'r', 's', 'a', '%'
	};
	// find conversion specifiers
	std::vector<FormatSpec> conversions;
	const auto codepoints = this->codepoints();
	for (size_t i = 0, start = 0; i < codepoints.size(); ++i) {
		auto codepoint = codepoints[i];
		size_t end = start + utf8::codepoint_length(codepoint);
		if (codepoint == '%') {
			auto &conversion = conversions.emplace_back();
			codepoint = codepoints[++i];
			end += utf8::codepoint_length(codepoint);
			if (codepoint == '(') {
				// mapping
				TODO();
				codepoint = codepoints[++i];
				end += utf8::codepoint_length(codepoint);
			}
			if (codepoint == '#' || codepoint == '0' || codepoint == '-' || codepoint == ' '
				|| codepoint == '+') {
				// conversion flag
				TODO();
				codepoint = codepoints[++i];
				end += utf8::codepoint_length(codepoint);
			}
			if ((codepoint >= '0' && codepoint <= '9') || codepoint == '*') {
				// Minimum field width flag
				TODO();
				codepoint = codepoints[++i];
				end += utf8::codepoint_length(codepoint);
			}
			if (codepoint == '.') {
				// precision
				TODO();
				codepoint = codepoints[++i];
				end += utf8::codepoint_length(codepoint);
			}
			if (codepoint == 'h' || codepoint == 'l' || codepoint == 'L') {
				// A length modifier (h, l, or L) may be present, but is ignored as it is not
				// necessary for Python – so e.g. %ld is identical to %d.
				codepoint = codepoints[++i];
				end += utf8::codepoint_length(codepoint);
			}
			if (codepoint > 127) {
				return Err(
					value_error("unsupported format character '?' ({}) at index {}", codepoint, i));
			}
			auto conversion_type_it = std::find(
				conversion_types.begin(), conversion_types.end(), static_cast<char>(codepoint));
			if (conversion_type_it == conversion_types.end()) {
				return Err(value_error("unsupported format character '{}' ({}) at index {}",
					static_cast<char>(codepoint),
					codepoint,
					i));
			}
			conversion.conversion_type = *conversion_type_it;
			conversion.start = start;
			conversion.end = end;
		}
		start = end;
	}

	std::string new_value;
	new_value.reserve(static_cast<size_t>(static_cast<double>(m_value.size()) * 1.5));

	if (!values->type()->issubclass(types::tuple())) {
		auto values_ = PyTuple::create(const_cast<PyObject *>(values));
		if (values_.is_err()) { return Err(values_.unwrap_err()); }
		values = values_.unwrap();
	}

	const auto *tuple = static_cast<const PyTuple *>(values);
	if (tuple->size() != conversions.size()) {
		return Err(type_error("not enough arguments for format string"));
	}
	size_t start = 0;
	for (size_t index = 0; const auto &el : tuple->elements()) {
		const size_t end = conversions[index].start;
		ASSERT(end >= start);
		new_value.append(m_value.substr(start, end - start));
		auto obj_ = PyObject::from(el);
		if (obj_.is_err()) { return Err(obj_.unwrap_err()); }
		const auto conversion = conversions[index].apply(obj_.unwrap());
		if (conversion.is_err()) { return Err(conversion.unwrap_err()); }
		new_value.append(conversion.unwrap());
		start = conversions[index].end;
		index++;
	}
	new_value.append(m_value.substr(start));
	new_value.shrink_to_fit();
	return PyString::create(new_value);
}

std::optional<PyString::ReplacementField::Conversion> PyString::ReplacementField::get_conversion(
	char c)
{
	if (c == 'r') {
		return PyString::ReplacementField::Conversion::REPR;
	} else if (c == 's') {
		return PyString::ReplacementField::Conversion::STR;
	} else if (c == 'a') {
		return PyString::ReplacementField::Conversion::ASCII;
	}
	return std::nullopt;
}

namespace {
	PyResult<PyString::ReplacementField>
		parse_format(std::string_view str, size_t start, size_t end)
	{
		// replacement_field ::=  "{" [field_name] ["!" conversion] [":" format_spec] "}"
		// field_name        ::=  arg_name ("." attribute_name | "[" element_index "]")*
		// arg_name          ::=  [identifier | digit+]
		// attribute_name    ::=  identifier
		// element_index     ::=  digit+ | index_string
		// index_string      ::=  <any source character except "]"> +
		// conversion        ::=  "r" | "s" | "a"
		// format_spec       ::=  <described in the next section>
		ASSERT(str.front() == '{');
		ASSERT(str.back() == '}');
		PyString::ReplacementField replacement_field{ .start = start, .end = end };
		str = str.substr(1, str.size() - 2);
		if (!str.empty() && str[0] != '!' && str[0] != ':') {
			auto end = str.find_first_of("!:");
			replacement_field.field_name = str.substr(0, end);
			if (end == std::string_view::npos) { return Ok(replacement_field); }
			str = end == std::string_view::npos ? "" : str.substr(end);
		}
		if (!str.empty() && str[0] == '!') {
			str = str.substr(1);
			auto end = str.find(':');
			const auto conversion = PyString::ReplacementField::get_conversion(str.front());
			if (conversion.has_value()) {
				replacement_field.conversion = *conversion;
			} else {
				return Err(value_error("Invalid conversion specifier '{}'", *conversion));
			}
			str = end == std::string_view::npos ? "" : str.substr(end);
		}
		if (!str.empty() && str[0] == ':') {
			return Err(not_implemented_error("Format spec in str.format not implemented"));
		}
		ASSERT(str.empty());
		return Ok(replacement_field);
	}
}// namespace

PyResult<PyObject *> PyString::format(PyTuple *args, PyDict *kwargs) const
{
	std::string_view str{ m_value };
	std::string new_string;
	const auto &cps = codepoints();
	size_t start = 0;
	size_t index = 0;
	size_t args_index = 0;
	auto increment = [&start, &index, &cps, this](size_t count) {
		while (count-- > 0) {
			ASSERT(index < cps.size());
			const auto cp = cps[index++];
			ASSERT(start < m_value.size());
			start += utf8::codepoint_length(cp);
		}
	};
	for (; index < cps.size(); increment(1)) {
		const auto cp = cps[index];
		if (cp == '{') {
			if (cps[index + 1] == '{') {
				increment(1);
				new_string.push_back('{');
				continue;
			}
			const auto it = std::find(cps.begin() + start, cps.end(), '}');
			if (it == cps.end()) {
				return Err(value_error("Single '{{' encountered in format string"));
			}
			const auto end = std::distance(cps.begin() + start, it) + 1;

			auto replacement_field_ = parse_format(str.substr(start, end), start, start + end);
			if (replacement_field_.is_err()) { return Err(replacement_field_.unwrap_err()); }

			const auto &replacement_field = replacement_field_.unwrap();
			auto r = [args, kwargs, &args_index, &replacement_field, &str, start, end]()
				-> PyResult<PyString *> {
				return [args, kwargs, &args_index, &replacement_field]()
						   -> PyResult<PyObject *> {
					if (replacement_field.field_name.has_value()) {
						if (!kwargs) { return Err(key_error(*replacement_field.field_name)); }
						const auto it = kwargs->map().find(String{ *replacement_field.field_name });
						if (it != kwargs->map().end()) { return PyObject::from(it->second); }
						return Err(key_error(*replacement_field.field_name));
					} else {
						if (!args || args_index >= args->elements().size()) {
							return Err(index_error(
								"Replacement index {} out of range for positional args tuple",
								args_index));
						}
						return PyObject::from(args->elements()[args_index++]);
					}
				}()
								  .and_then(
									  [&replacement_field](PyObject *obj) -> PyResult<PyString *> {
										  if (replacement_field.conversion.has_value()) {
											  switch (*replacement_field.conversion) {
											  case ReplacementField::Conversion::REPR: {
												  return obj->repr();
											  } break;
											  case ReplacementField::Conversion::STR: {
												  return obj->str();
											  } break;
											  case ReplacementField::Conversion::ASCII: {
												  return PyString::convert_to_ascii(obj);
											  } break;
											  }
										  }
										  return obj->str();
									  })
								  .and_then([&replacement_field, &str, start, end](
												PyString *obj) -> PyResult<PyString *> {
									  if (replacement_field.format_spec.has_value()) {
										  return Err(not_implemented_error(
											  "Replacement field '{}' not implemented",
											  str.substr(start + 1, end - 2)));
									  }
									  return Ok(obj);
								  });
			}()
					   .and_then([&new_string](PyString *stringified) -> PyResult<std::monostate> {
						   new_string += stringified->value();
						   return Ok(std::monostate{});
					   });

			if (r.is_err()) { return Err(r.unwrap_err()); }
			increment(replacement_field.end - start - 1);
		} else {
			new_string += str.substr(start, utf8::codepoint_length(cp));
		}
	}

	return create(new_string);
}

PyResult<PyObject *> PyString::maketrans(PyTuple *args, PyDict *kwargs)
{
	auto parse_result = PyArgsParser<PyObject *, PyObject *, PyObject *>::unpack_tuple(args,
		kwargs,
		"str.maketrans",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 3>{},
		nullptr,
		nullptr);

	if (parse_result.is_err()) { return Err(parse_result.unwrap_err()); }
	auto [x, y, z] = parse_result.unwrap();
	if (y || z) {
		return Err(
			not_implemented_error("str.maketrans with two or three arguments is not implemented"));
	}
	if (!x->type()->issubclass(types::dict())) {
		return Err(type_error("if you give only one argument to maketrans it must be a dict"));
	}

	const auto *x_dict = static_cast<PyDict *>(x);
	auto result_ = PyDict::create();
	if (result_.is_err()) { return result_; }
	auto *result = result_.unwrap();
	for (const auto &[key, value] : x_dict->map()) {
		PyObject *key_obj = nullptr;
		PyObject *value_obj = nullptr;
		auto key_ = PyObject::from(key);
		if (key_.is_err()) { return key_; }
		if (key_.unwrap()->type()->issubclass(types::integer())) {
			key_obj = key_.unwrap();
		} else if (key_.unwrap()->type()->issubclass(types::str())) {
			auto *key_str = static_cast<const PyString *>(key_.unwrap());
			const auto cp = key_str->codepoint();
			if (!cp.has_value()) {
				return Err(value_error(" string keys in translate table must be of length 1"));
			}
			key_obj = PyInteger::create(*cp).unwrap();
		} else {
			return Err(type_error("keys in translate table must be strings or integers"));
		}

		auto value_ = PyObject::from(value);
		if (value_.is_err()) { return value_; }
		if (value_.unwrap()->type()->issubclass(types::integer())
			|| value_.unwrap()->type()->issubclass(types::str()) || value_.unwrap() == py_none()) {
			value_obj = value_.unwrap();
		} else {
			return Err(type_error("keys in translate table must be strings or integers"));
		}

		ASSERT(key_obj);
		ASSERT(value_obj);
		result->insert(key_obj, value_obj);
	}

	return Ok(result);
}

PyResult<PyObject *> PyString::isidentifier() const
{
	for (size_t i = 0; i < m_value.size();) {
		const auto length = utf8::codepoint_length(m_value[i]);
		const auto cp = utf8::codepoint(m_value.c_str() + i, length);
		ASSERT(cp.has_value());
		if (i == 0 && !u_hasBinaryProperty(*cp, UProperty::UCHAR_XID_START)) {
			return Ok(py_false());
		} else if (!u_hasBinaryProperty(*cp, UProperty::UCHAR_XID_CONTINUE)) {
			return Ok(py_false());
		}
		i += length;
	}
	return Ok(py_true());
}

PyResult<PyString *> PyString::convert_to_ascii(PyObject *obj)
{
	return obj->repr().and_then([obj](PyString *str) {
		std::string new_string;
		const auto &cps = str->codepoints();
		auto s = [obj, &cps]() {
			if (compare_slot_address(obj->type()->underlying_type().__repr__,
					types::str()->underlying_type().__repr__)) {
				// if we got the representation with str.__repr__, remove the surrounding single
				// quotes
				return std::span{ cps.begin() + 1, cps.size() - 2 };
			}
			return std::span{ cps.begin(), cps.size() };
		}();
		for (const auto &cp : s) {
			if (cp <= 128) {
				new_string.push_back(static_cast<char>(cp));
			} else {
				new_string += fmt::format("\\U{:08x}", cp);
			}
		}

		return PyString::create(new_string);
	});
}

PyResult<PyString *> PyString::from_encoded_object(const PyObject *obj,
	const std::string &encoding,
	const std::string &errors)
{
	ASSERT(obj);

	if (obj->type()->issubclass(types::bytes())) {
		if (!encoding.empty() && encoding != "utf-8") {
			return Err(not_implemented_error(
				"only utf-8 encoding implemented for 'str' decoding, got {}", encoding));
		}
		if (static_cast<const PyBytes &>(*obj).value().b.empty()) { return PyString::create(""); }

		return PyString::decode(
			std::span<const std::byte>{ static_cast<const PyBytes &>(*obj).value().b.begin(),
				static_cast<const PyBytes &>(*obj).value().b.end() },
			encoding,
			errors);
	}
	return Err(not_implemented_error("PyString::from_encoded_object only implemented for 'bytes'"));
}

PyResult<PyString *> PyString::decode(std::span<const std::byte> bytes,
	const std::string &encoding,
	const std::string & /*errors*/)
{
	if (encoding.empty() || encoding == "utf-8") {
		icu::UnicodeString uni_str{ static_cast<int32_t>(bytes.size()), UChar32{}, 0 };
		std::string encoded_cp;
		encoded_cp.reserve(4);
		while (!bytes.empty()) {
			auto c = bytes.front();
			ASSERT(
				static_cast<int>(c) < static_cast<int>(std::numeric_limits<unsigned char>::max()));
			auto size = utf8::codepoint_length(static_cast<char>(bytes.front()));
			if (size > bytes.size()) {
				return Err(value_error("str.decode: malformed utf-8 sequence"));
			}
			for (const auto &el : bytes.subspan(0, bytes.size())) {
				ASSERT(static_cast<int>(el)
					   < static_cast<int>(std::numeric_limits<unsigned char>::max()));
				encoded_cp.push_back(static_cast<char>(el));
			}
			const auto cp = utf8::codepoint(encoded_cp.data(), size);
			if (!cp.has_value()) {
				return Err(value_error("invalid utf-8 encoded codepoint {}", encoded_cp));
			}
			encoded_cp.clear();
			uni_str.append(UChar32{ static_cast<int32_t>(*cp) });
			bytes = bytes.subspan(size);
		}
		std::string result;
		result.reserve(uni_str.length() * 2);
		uni_str.toUTF8String(result);
		return PyString::create(std::move(result));
	}

	return Err(not_implemented_error("str.decode only implemented for 'utf-8' encoding"));
}

PyResult<PyString *> PyString::chr(BigIntType cp)
{
	if (cp < 0 || cp >= 0x110000) { return Err(value_error("chr() arg not in range(0x110000)")); }

	ASSERT(cp.fits_uint_p());

	icu::UnicodeString s;
	s.append(UChar32{ static_cast<int32_t>(cp.get_ui()) });
	std::string result;
	result.reserve(s.length() * 2);
	s.toUTF8String(result);
	return PyString::create(std::move(result));
}

PyResult<PyObject *> PyString::replace(PyTuple *args, PyDict *kwargs) const
{
	auto parse_result = PyArgsParser<PyString *, PyString *, PyInteger *>::unpack_tuple(args,
		kwargs,
		"str.find",
		std::integral_constant<size_t, 2>{},
		std::integral_constant<size_t, 3>{},
		nullptr);

	if (parse_result.is_err()) { return Err(parse_result.unwrap_err()); }

	auto [old, new_, count] = parse_result.unwrap();

	if (old->value().empty() && new_->value().empty()) { return PyString::create(m_value); }
	if (count && count->as_big_int() == 0) { return PyString::create(m_value); }

	const size_t count_ = [count]() {
		if (!count || count->as_big_int() < 0) { return std::numeric_limits<size_t>::max(); }
		return count->as_size_t();
	}();

	size_t counter = 0;
	icu::UnicodeString result;
	const auto cps = codepoints();
	if (old->value().empty()) {
		for (auto it = cps.begin(); it != cps.end(); ++it, ++counter) {
			if (counter < count_) {
				for (const auto &el : new_->value()) { result.append(UChar32{ el }); }
			}
			result.append(UChar32(*it));
		}
		if (counter < count_) {
			for (const auto &el : new_->value()) { result.append(UChar32{ el }); }
		}
	} else {
		const auto old_cps = old->codepoints();

		for (auto it = cps.begin(); it != cps.end();) {
			if (old_cps.size() > static_cast<size_t>(std::distance(it, cps.end()))) {
				while (it != cps.end()) {
					result.append(UChar32(*it));
					++it;
				}
				break;
			}
			std::span next{ it, it + old_cps.size() };
			if (counter < count_
				&& std::equal(next.begin(), next.end(), old_cps.begin(), old_cps.end())) {
				for (const auto &el : new_->value()) { result.append(el); }
				it += old_cps.size();
				counter++;
			} else {
				result.append(UChar32(*it));
				++it;
			}
		}
	}

	std::string result_str;
	result.toUTF8String(result_str);
	return PyString::create(result_str);
}

namespace {

	std::once_flag str_flag;

	std::unique_ptr<TypePrototype> register_string()
	{
		return std::move(klass<PyString>("str")
							 .def("isalnum", &PyString::isalnum)
							 .def("isalpha", &PyString::isalpha)
							 .def("isascii", &PyString::isascii)
							 .def("isdigit", &PyString::isdigit)
							 .def("islower", &PyString::islower)
							 .def("isupper", &PyString::isupper)
							 .def("isidentifier", &PyString::isidentifier)
							 .def("capitalize", &PyString::capitalize)
							 .def("casefold", &PyString::casefold)
							 .def("count", &PyString::count)
							 .def("startswith", &PyString::startswith)
							 .def("endswith", &PyString::endswith)
							 .def("find", &PyString::find)
							 .def("rfind", &PyString::rfind)
							 .def("join", &PyString::join)
							 .def("lower", &PyString::lower)
							 .def("upper", &PyString::upper)
							 .def("rpartition", &PyString::rpartition)
							 .def("strip", &PyString::strip)
							 .def("rstrip", &PyString::rstrip)
							 .def("split", &PyString::split)
							 .def("format", &PyString::format)
							 .def("replace", &PyString::replace)
							 .staticmethod("maketrans", &PyString::maketrans)
							 .type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyString::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(str_flag, []() { type = register_string(); });
		return std::move(type);
	};
}

PyStringIterator::PyStringIterator(const PyString &pystring)
	: PyBaseObject(types::BuiltinTypes::the().str_iterator()), m_pystring(pystring)
{}

std::string PyStringIterator::to_string() const
{
	return fmt::format("<str_iterator at {}>", static_cast<const void *>(this));
}

void PyStringIterator::visit_graph(Visitor &visitor)
{
	PyObject::visit_graph(visitor);
	visitor.visit(const_cast<PyString &>(m_pystring));
}

PyResult<PyObject *> PyStringIterator::__repr__() const { return PyString::create(to_string()); }

PyResult<PyObject *> PyStringIterator::__next__()
{
	if (m_current_index < m_pystring.size()) {
		const auto cp = m_pystring.codepoints()[m_current_index++];
		ASSERT(cp < 0x110000);
		icu::UnicodeString uni_str(static_cast<UChar32>(cp));
		std::string str;
		uni_str.toUTF8String(str);
		return PyString::create(str);
	}
	return Err(stop_iteration());
}

PyType *PyStringIterator::static_type() const { return types::str_iterator(); }

namespace {

	std::once_flag str_iterator_flag;

	std::unique_ptr<TypePrototype> register_str_iterator()
	{
		return std::move(klass<PyStringIterator>("str_iterator").type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyStringIterator::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(str_iterator_flag, []() { type = register_str_iterator(); });
		return std::move(type);
	};
}

}// namespace py
