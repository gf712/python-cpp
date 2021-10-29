#include "PyString.hpp"

#include "interpreter/Interpreter.hpp"


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
	if (u0 >= 0 && u0 <= 127) return u0;
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

PyObject *PyString::add_impl(const PyObject *obj, Interpreter &interpreter) const
{
	if (auto rhs = as<PyString>(obj)) {
		return PyString::create(m_value + rhs->value());
	} else {

		interpreter.raise_exception(
			"TypeError: unsupported operand type(s) for +: \'{}\' and \'{}\'",
			object_name(type()),
			object_name(obj->type()));
		return nullptr;
	}
}


PyString::PyString(std::string s) : PyObject(PyObjectType::PY_STRING), m_value(std::move(s))
{
	m_slots.hash = [this]() { return this->hash_impl(*VirtualMachine::the().interpreter()); };
	m_slots.richcompare = [this](const PyObject *other, RichCompare op) {
		return this->richcompare_impl(other, op, *VirtualMachine::the().interpreter());
	};
}

size_t PyString::hash_impl(Interpreter &) const { return std::hash<std::string>{}(m_value); }

PyObject *PyString::repr_impl(Interpreter &) const { return PyString::from(String{ m_value }); }

PyObject *PyString::equal_impl(const PyObject *obj, Interpreter &interpreter) const
{
	if (auto obj_string = as<PyString>(obj)) {
		return m_value == obj_string->value() ? py_true() : py_false();
	} else {
		return PyObject::equal_impl(obj, interpreter);
	}
}


PyObject *PyString::richcompare_impl(const PyObject *other,
	RichCompare op,
	Interpreter &interpreter) const
{
	spdlog::debug("PyString::richcompare_impl: Compare {} to {} using {} op",
		to_string(),
		other->to_string(),
		static_cast<int>(op));
	if (auto obj_string = as<PyString>(other)) {
		switch (op) {
		case RichCompare::Py_LT: {
			return m_value < obj_string->value() ? py_true() : py_false();
		}
		case RichCompare::Py_LE: {
			return m_value <= obj_string->value() ? py_true() : py_false();
		}
		case RichCompare::Py_EQ: {
			if (this == obj_string) return py_true();
			return m_value == obj_string->value() ? py_true() : py_false();
		}
		case RichCompare::Py_NE: {
			return m_value != obj_string->value() ? py_true() : py_false();
		}
		case RichCompare::Py_GT: {
			return m_value > obj_string->value() ? py_true() : py_false();
		}
		case RichCompare::Py_GE: {
			return m_value >= obj_string->value() ? py_true() : py_false();
		}
		}
		return m_value == obj_string->value() ? py_true() : py_false();
	} else {
		return PyObject::richcompare_impl(other, op, interpreter);
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
