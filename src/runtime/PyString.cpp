#include "PyString.hpp"
#include "PyList.hpp"
#include "PyNumber.hpp"
#include "TypeError.hpp"

#include "interpreter/Interpreter.hpp"
#include "types/api.hpp"

#include <numeric>

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

PyString *PyString::create(const std::string &value)
{
	auto &heap = VirtualMachine::the().heap();
	return heap.allocate<PyString>(value);
}

// PyString *PyString::create(PyString *self, PyTuple *args, PyDict *kwargs)
// {
// 	// FIXME with proper error handling
// 	ASSERT(!args)
// 	ASSERT(!kwargs)

// 	return self;
// }

PyObject *PyString::add_impl(const PyObject *obj) const
{
	if (auto rhs = as<PyString>(obj)) {
		return PyString::create(m_value + rhs->value());
	} else {
		VirtualMachine::the().interpreter().raise_exception(
			"TypeError: unsupported operand type(s) for +: \'{}\' and \'{}\'",
			object_name(type()),
			object_name(obj->type()));
		return py_none();
	}
}


PyString::PyString(std::string s) : PyBaseObject(PyObjectType::PY_STRING), m_value(std::move(s))
{
	m_slots->hash = [this]() { return this->hash_impl(VirtualMachine::the().interpreter()); };
	m_slots->richcompare = [this](const PyObject *other, RichCompare op) {
		return this->richcompare_impl(other, op, VirtualMachine::the().interpreter());
	};
}

size_t PyString::hash_impl(Interpreter &) const { return std::hash<std::string>{}(m_value); }

PyObject *PyString::repr_impl() const { return PyString::create(m_value); }


PyObject *PyString::equal_impl(const PyObject *obj, Interpreter &) const
{
	if (this == obj) return py_true();
	if (auto obj_string = as<PyString>(obj)) {
		return m_value == obj_string->value() ? py_true() : py_false();
	} else {
		type_error("'==' not supported between instances of '{}' and '{}'",
			object_name(type()),
			object_name(obj->type()));
		return nullptr;
	}
}

PyObject *PyString::less_than_impl(const PyObject *obj, Interpreter &) const
{
	if (this == obj) return py_true();
	if (auto obj_string = as<PyString>(obj)) {
		return m_value < obj_string->value() ? py_true() : py_false();
	} else {
		type_error("'==' not supported between instances of '{}' and '{}'",
			object_name(type()),
			object_name(obj->type()));
		return nullptr;
	}
}


PyObject *PyString::len_impl(Interpreter &) const
{
	size_t size{ 0 };
	for (auto it = m_value.begin(); it != m_value.end();) {
		const auto codepoint_byte_size = utf8::codepoint_length(*it);
		size++;
		it += codepoint_byte_size;
	}
	return PyObject::from(Number{ static_cast<int64_t>(size) });
}

// FIXME: assumes string only has ASCII characters
PyString *PyString::capitalize() const
{
	auto new_string = m_value;
	new_string[0] = std::toupper(new_string[0]);
	return PyString::create(new_string);
}

// FIXME: assumes string only has ASCII characters
PyString *PyString::casefold() const
{
	auto new_string = m_value;
	std::transform(new_string.begin(),
		new_string.end(),
		new_string.begin(),
		[](const unsigned char c) -> unsigned char { return std::tolower(c); });
	return PyString::create(new_string);
}

// FIXME: assumes string only has ASCII characters
PyObject *PyString::isalnum() const
{
	if (m_value.empty()) { return py_false(); }

	auto it = std::find_if_not(
		m_value.begin(), m_value.end(), [](const unsigned char c) { return std::isalnum(c); });
	if (it == m_value.end()) {
		return py_true();
	} else {
		return py_false();
	}
}

// FIXME: assumes string only has ASCII characters
PyObject *PyString::isalpha() const
{
	if (m_value.empty()) { return py_false(); }

	auto it = std::find_if_not(
		m_value.begin(), m_value.end(), [](const unsigned char c) { return std::isalpha(c); });
	if (it == m_value.end()) {
		return py_true();
	} else {
		return py_false();
	}
}

// FIXME: assumes string only has ASCII characters
PyObject *PyString::isdigit() const
{
	if (m_value.empty()) { return py_false(); }

	auto it = std::find_if_not(
		m_value.begin(), m_value.end(), [](const unsigned char c) { return std::isdigit(c); });
	if (it == m_value.end()) {
		return py_true();
	} else {
		return py_false();
	}
}

PyObject *PyString::isascii() const
{
	for (size_t i = 0; i < m_value.size();) {
		int length = utf8::codepoint_length(m_value[i]);
		const auto codepoint = utf8::codepoint(m_value.c_str(), length);
		if (codepoint > 0x7F) { return py_false(); }
		i += length;
	}
	return py_true();
}

// FIXME: assumes string only has ASCII characters
PyObject *PyString::islower() const
{
	if (m_value.empty()) { return py_false(); }

	auto it = std::find_if_not(m_value.begin(), m_value.end(), [](const unsigned char c) {
		return !std::isalpha(c) || std::islower(c);
	});
	if (it == m_value.end()) {
		return py_true();
	} else {
		return py_false();
	}
}

// FIXME: assumes string only has ASCII characters
PyObject *PyString::isupper() const
{
	if (m_value.empty()) { return py_false(); }

	auto it = std::find_if_not(m_value.begin(), m_value.end(), [](const unsigned char c) {
		return !std::isalpha(c) || std::isupper(c);
	});
	if (it == m_value.end()) {
		return py_true();
	} else {
		return py_false();
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
PyNumber *PyString::find(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(args && args->size() <= 3 && args->size() > 0)
	ASSERT(!kwargs)

	PyString *pattern = as<PyString>(PyObject::from(args->elements()[0]));
	PyNumber *start = nullptr;
	PyNumber *end = nullptr;
	size_t result{ std::string::npos };

	if (args->size() >= 2) {
		start = as<PyNumber>(PyObject::from(args->elements()[1]));
		// TODO: raise exception when start in not a number
		ASSERT(start)
	}
	if (args->size() == 3) {
		end = as<PyNumber>(PyObject::from(args->elements()[2]));
		// TODO: raise exception when end in not a number
		ASSERT(end)
	}

	if (!start && !end) {
		result = m_value.find(pattern->value().c_str());
	} else if (!end) {
		size_t start_ = std::visit(
			[this](const auto &val) { return get_position_from_slice(static_cast<int64_t>(val)); },
			start->value().value);
		result = m_value.find(pattern->value().c_str(), start_);
	} else {
		size_t start_ = std::visit(
			[this](const auto &val) { return get_position_from_slice(static_cast<int64_t>(val)); },
			start->value().value);
		size_t end_ = std::visit(
			[this](const auto &val) { return get_position_from_slice(static_cast<int64_t>(val)); },
			end->value().value);
		size_t subtring_size = end_ - start_;
		result = m_value.find(pattern->value().c_str(), start_, subtring_size);
	}
	if (result == std::string::npos) {
		return PyNumber::create(Number{ int64_t{ -1 } });
	} else {
		return PyNumber::create(Number{ static_cast<int64_t>(result) });
	}
}

// FIXME: assumes string only has ASCII characters
PyNumber *PyString::count(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(args && args->size() <= 3 && args->size() > 0)
	ASSERT(!kwargs)

	PyString *pattern = as<PyString>(PyObject::from(args->elements()[0]));
	PyNumber *start = nullptr;
	PyNumber *end = nullptr;
	size_t result{ 0 };

	if (args->size() >= 2) {
		start = as<PyNumber>(PyObject::from(args->elements()[1]));
		// TODO: raise exception when start in not a number
		ASSERT(start)
	}
	if (args->size() == 3) {
		end = as<PyNumber>(PyObject::from(args->elements()[2]));
		// TODO: raise exception when end in not a number
		ASSERT(end)
	}

	const size_t start_ = [start, this]() {
		if (start) {
			return std::visit(
				[this](
					const auto &val) { return get_position_from_slice(static_cast<int64_t>(val)); },
				start->value().value);
		} else {
			return size_t{ 0 };
		}
	}();

	const size_t end_ = [end, this]() {
		if (end) {
			return std::visit(
				[this](
					const auto &val) { return get_position_from_slice(static_cast<int64_t>(val)); },
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

	return PyNumber::create(Number{ static_cast<int64_t>(result) });
}

// FIXME: assumes string only has ASCII characters
PyObject *PyString::endswith(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(args && args->size() <= 3 && args->size() > 0)
	ASSERT(!kwargs)

	PyString *suffix = as<PyString>(PyObject::from(args->elements()[0]));
	PyNumber *start = nullptr;
	PyNumber *end = nullptr;
	bool result{ false };

	if (args->size() >= 2) {
		start = as<PyNumber>(PyObject::from(args->elements()[1]));
		// TODO: raise exception when start in not a number
		ASSERT(start)
	}
	if (args->size() == 3) {
		end = as<PyNumber>(PyObject::from(args->elements()[2]));
		// TODO: raise exception when end in not a number
		ASSERT(end)
	}

	if (!start && !end) {
		result = m_value.ends_with(suffix->value());
	} else if (!end) {
		size_t start_ = std::visit(
			[this](const auto &val) { return get_position_from_slice(static_cast<int64_t>(val)); },
			start->value().value);
		std::string_view substring{ m_value.c_str() + start_, m_value.size() - start_ };
		result = substring.ends_with(suffix->value());
	} else {
		size_t start_ = std::visit(
			[this](const auto &val) { return get_position_from_slice(static_cast<int64_t>(val)); },
			start->value().value);
		size_t end_ = std::visit(
			[this](const auto &val) { return get_position_from_slice(static_cast<int64_t>(val)); },
			end->value().value);
		std::string_view substring{ m_value.c_str() + start_, end_ - start_ };
		result = substring.ends_with(suffix->value());
	}

	return result ? py_true() : py_false();
}

PyString *PyString::join(PyTuple *args, PyDict *kwargs) const
{
	ASSERT(args && args->size() == 1)
	ASSERT(!kwargs)

	auto *string_list = as<PyList>(PyObject::from(args->elements()[0]));

	if (string_list->elements().empty()) { return PyString::create(""); }

	std::string begin = std::visit(overloaded{ [](const auto &) -> std::string {
												  // TODO: should raise -> join only works with
												  // strings
												  TODO()
												  return {};
											  },
									   [](const String &value) { return value.s; },
									   [](PyObject *value) {
										   // TODO: should raise -> join only works with strings
										   ASSERT(as<PyString>(value))
										   return as<PyString>(value)->value();
									   } },
		string_list->elements()[0]);

	std::string result = std::accumulate(std::next(string_list->elements().begin()),
		string_list->elements().end(),
		begin,
		[this](std::string lhs, const Value &rhs) {
			return std::move(lhs) + m_value
				   + std::visit(overloaded{ [](const auto &) -> std::string {
											   // TODO: should raise -> join only works with
											   // strings
											   TODO()
											   return {};
										   },
									[](const String &value) { return value.s; },
									[](PyObject *value) {
										// TODO: should raise -> join only works with strings
										ASSERT(as<PyString>(value))
										return as<PyString>(value)->value();
									} },
					   rhs);
		});

	return PyString::create(result);
}

// FIXME: assumes string only has ASCII characters
PyString *PyString::lower() const
{
	auto new_string = m_value;
	std::transform(new_string.begin(),
		new_string.end(),
		new_string.begin(),
		[](const unsigned char c) -> unsigned char { return std::tolower(c); });
	return PyString::create(new_string);
}

// FIXME: assumes string only has ASCII characters
PyString *PyString::upper() const
{
	auto new_string = m_value;
	std::transform(new_string.begin(),
		new_string.end(),
		new_string.begin(),
		[](const unsigned char c) -> unsigned char { return std::toupper(c); });
	return PyString::create(new_string);
}


std::vector<int32_t> PyString::codepoints() const
{
	std::vector<int32_t> codepoints;

	for (size_t i = 0; i < m_value.size();) {
		int length = utf8::codepoint_length(m_value[i]);
		codepoints.push_back(utf8::codepoint(m_value.c_str(), length));
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

void PyString::register_type(PyModule *m)
{
	klass<PyString>(m, "str")
		.def("isalnum", &PyString::isalnum)
		.def("isalpha", &PyString::isalpha)
		.def("isascii", &PyString::isascii)
		.def("isdigit", &PyString::isdigit)
		.def("islower", &PyString::islower)
		.def("isupper", &PyString::isupper)
		.def("capitalize", &PyString::capitalize)
		.def("casefold", &PyString::casefold)
		.def("count", &PyString::count)
		.def("endswith", &PyString::endswith)
		.def("find", &PyString::find)
		.def("join", &PyString::join)
		.def("lower", &PyString::lower)
		.def("upper", &PyString::upper);
}
