#include "PyString.hpp"
#include "IndexError.hpp"
#include "MemoryError.hpp"
#include "NotImplementedError.hpp"
#include "PyBool.hpp"
#include "PyDict.hpp"
#include "PyInteger.hpp"
#include "PyList.hpp"
#include "PyNone.hpp"
#include "PySlice.hpp"
#include "StopIteration.hpp"
#include "TypeError.hpp"
#include "ValueError.hpp"
#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"
#include "types/builtin.hpp"

#include <mutex>
#include <numeric>

#include <unicode/unistr.h>

namespace py {

template<> PyString *as(PyObject *obj)
{
	if (obj->type() == str()) { return static_cast<PyString *>(obj); }
	return nullptr;
}

template<> const PyString *as(const PyObject *obj)
{
	if (obj->type() == str()) { return static_cast<const PyString *>(obj); }
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

	size_t codepoint_length(int32_t codepoint)
	{
		if (codepoint <= 0x00007F) { return 1; }
		if (codepoint <= 0x0007FF) { return 2; }
		if (codepoint <= 0x00FFFF) { return 3; }
		ASSERT(codepoint <= 0x10FFFF);
		return 4;
	}

	// take from http://www.zedwood.com/article/cpp-utf8-char-to-codepoint
	int32_t codepoint(const char *str, size_t length)
	{
		if (length < 1) return -1;
		unsigned char u0 = str[0];
		if (u0 <= 127) return u0;
		if (length < 2) return -1;
		unsigned char u1 = str[1];
		if (u0 >= 192 && u0 <= 223) return (u0 - 192) * 64 + (u1 - 128);
		if (u0 == 0xed && (u1 & 0xa0) == 0xa0) return -1;// code points, 0xd800 to 0xdfff
		if (length < 3) return -1;
		unsigned char u2 = str[2];
		if (u0 >= 224 && u0 <= 239) return (u0 - 224) * 4096 + (u1 - 128) * 64 + (u2 - 128);
		if (length < 4) return -1;
		unsigned char u3 = str[3];
		if (u0 >= 240 && u0 <= 247)
			return (u0 - 240) * 262144 + (u1 - 128) * 4096 + (u2 - 128) * 64 + (u3 - 128);
		return -1;
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
		return PyString::create(s->value());
	} else if (obj->type()->underlying_type().__str__.has_value()) {
		return obj->str();
	} else {
		return obj->repr().and_then([](PyObject *str) -> PyResult<PyString *> {
			if (!as<PyString>(str)) {
				return Err(
					type_error("__repr_ returned non-string (type {})", str->type()->name()));
			}
			return Ok(as<PyString>(str));
		});
	}
}

PyResult<PyObject *> PyString::__new__(const PyType *type, PyTuple *args, PyDict *kwargs)
{
	// FIXME: this should use either __str__ or __repr__ rather than relying on first arg being a
	// String
	// FIXME: handle bytes_or_buffer argument
	// FIXME: handle encoding argument
	// FIXME: handle errors argument
	ASSERT(!kwargs || kwargs->map().size() == 0)
	ASSERT(args && args->size() == 1)
	ASSERT(type == py::str())

	const auto &string = args->elements()[0];
	if (std::holds_alternative<String>(string)) {
		return PyString::create(std::get<String>(string).s);
	} else if (std::holds_alternative<PyObject *>(string)) {
		auto s = std::get<PyObject *>(string);
		ASSERT(as<PyString>(s));
		return PyString::create(as<PyString>(s)->value());
	} else {
		TODO();
	}
}

PyString::PyString(std::string s) : PyBaseObject(BuiltinTypes::the().str()), m_value(std::move(s))
{}

PyResult<int64_t> PyString::__hash__() const
{
	return Ok(static_cast<int64_t>(std::hash<std::string>{}(m_value)));
}

PyResult<PyObject *> PyString::__repr__() const { return PyString::create(m_value); }

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
	new_string[0] = std::toupper(new_string[0]);
	return PyString::create(new_string);
}

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::casefold() const
{
	auto new_string = m_value;
	std::transform(new_string.begin(),
		new_string.end(),
		new_string.begin(),
		[](const unsigned char c) -> unsigned char { return std::tolower(c); });
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
		int length = utf8::codepoint_length(m_value[i]);
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
	while (start_idx != std::string::npos) {
		start_idx = m_value.find(pattern->value(), start_idx);
		if (start_idx != std::string::npos) { result = start_idx++; }
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

	auto suffix_ = PyObject::from(args->elements()[0]);
	if (suffix_.is_err()) return suffix_;
	PyString *suffix = as<PyString>(suffix_.unwrap());
	PyInteger *start = nullptr;
	PyInteger *end = nullptr;
	bool result{ false };

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
		result = m_value.ends_with(suffix->value());
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
		result = substring.ends_with(suffix->value());
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
		std::string_view substring{ m_value.c_str() + start_, end_ - start_ };
		result = substring.ends_with(suffix->value());
	}

	return Ok(result ? py_true() : py_false());
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
		[](const unsigned char c) -> unsigned char { return std::tolower(c); });
	return PyString::create(new_string);
}

// FIXME: assumes string only has ASCII characters
PyResult<PyObject *> PyString::upper() const
{
	auto new_string = m_value;
	std::transform(new_string.begin(),
		new_string.end(),
		new_string.begin(),
		[](const unsigned char c) -> unsigned char { return std::toupper(c); });
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

PyResult<PyObject *> PyString::rstrip(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(!kwargs || kwargs->size() == 0)

	const auto chars = [args]() -> PyResult<std::vector<int32_t>> {
		if (!args || args->size() == 0) { return Ok(std::vector<int32_t>{}); }
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


PyResult<PyObject *> PyString::format(PyTuple *, PyDict *) const
{
	// TODO: do some actual string formatting :)
	return create(m_value);
}

std::vector<int32_t> PyString::codepoints() const
{
	std::vector<int32_t> codepoints;

	for (size_t i = 0; i < m_value.size();) {
		int length = utf8::codepoint_length(m_value[i]);
		codepoints.push_back(utf8::codepoint(m_value.c_str() + i, length));
		i += length;
	}
	return codepoints;
}


std::optional<int32_t> PyString::codepoint() const
{
	if (auto codepoint_length = utf8::codepoint_length(m_value[0]);
		codepoint_length != m_value.size()) {
		return {};
	} else {
		return utf8::codepoint(m_value.c_str(), codepoint_length);
	}
}

PyType *PyString::static_type() const { return py::str(); }

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
			icu::UnicodeString uni_str(codepoints[idx]);
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

	icu::UnicodeString uni_str(codepoints()[index]);
	std::string str;
	uni_str.toUTF8String(str);
	return PyString::create(str);
}

namespace {
	struct Conversion
	{
		size_t start;
		size_t end;
		std::optional<std::string> mapping;
		std::optional<char> conversion_flag;
		std::optional<uint32_t> minimum_width;
		std::optional<uint32_t> precision;
		std::optional<char> conversion_type;

		PyResult<std::string> apply(PyObject *obj) const
		{
			if (mapping.has_value()) { TODO(); }
			if (conversion_flag.has_value()) { TODO(); }
			if (minimum_width.has_value()) { TODO(); }
			if (precision.has_value()) { TODO(); }
			if (mapping.has_value()) { TODO(); }
			ASSERT(conversion_type.has_value());
			switch (*conversion_type) {
			case 's': {
				return PyString::create(obj).and_then(
					[](PyString *str) { return Ok(str->value()); });
			} break;
			default:
				return Err(not_implemented_error(
					"printf conversion type '{}' not implemented", *conversion_type));
			}
		}
	};
}// namespace

PyResult<PyString *> PyString::printf(const PyObject *values) const
{
	static constexpr std::array conversion_types = {
		'd', 'i', 'o', 'u', 'x', 'X', 'e', 'E', 'f', 'F', 'g', 'G', 'c', 'r', 's', 'a', '%'
	};
	// find conversion specifiers
	std::vector<Conversion> conversions;
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
	new_value.reserve(static_cast<size_t>(m_value.size() * 1.5));

	if (auto tuple = as<PyTuple>(values)) {
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
	}
	new_value.shrink_to_fit();
	return PyString::create(new_value);
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
							 .def("rstrip", &PyString::rstrip)
							 .def("format", &PyString::format)
							 .type);
	}
}// namespace

std::function<std::unique_ptr<TypePrototype>()> PyString::type_factory()
{
	return [] {
		static std::unique_ptr<TypePrototype> type = nullptr;
		std::call_once(str_flag, []() { type = py::register_string(); });
		return std::move(type);
	};
}

PyStringIterator::PyStringIterator(const PyString &pystring)
	: PyBaseObject(BuiltinTypes::the().str_iterator()), m_pystring(pystring)
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
		icu::UnicodeString uni_str(m_pystring.codepoints()[m_current_index++]);
		std::string str;
		uni_str.toUTF8String(str);
		return PyString::create(str);
	}
	return Err(stop_iteration());
}

PyType *PyStringIterator::static_type() const { return str_iterator(); }

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
