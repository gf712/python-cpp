#include "Value.hpp"
#include "PyBool.hpp"
#include "PyBytes.hpp"
#include "PyEllipsis.hpp"
#include "PyFloat.hpp"
#include "PyInteger.hpp"
#include "PyNone.hpp"
#include "PyNumber.hpp"
#include "PyString.hpp"
#include "PyTuple.hpp"
#include "PyType.hpp"
#include "TypeError.hpp"
#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"

using namespace py;

std::string Number::to_string() const
{
	return std::visit([](const auto &value) { return fmt::format("{}", value); }, value);
}

Number Number::exp(const Number &rhs) const
{
	return std::visit(
		overloaded{
			[](const mpz_class &lhs_value, const mpz_class &rhs_value) -> Number {
				if (rhs_value.fits_ulong_p()) {
					mpz_class result{};
					mpz_pow_ui(result.get_mpz_t(), lhs_value.get_mpz_t(), rhs_value.get_ui());
					return Number{ std::move(result) };
				} else {
					return Number{ std::pow(lhs_value.get_d(), rhs_value.get_d()) };
				}
			},
			[](const mpz_class &lhs_value, const double &rhs_value) -> Number {
				return Number{ std::pow(lhs_value.get_d(), rhs_value) };
			},
			[](const double &lhs_value, const mpz_class &rhs_value) -> Number {
				return Number{ std::pow(lhs_value, rhs_value.get_d()) };
			},
			[](const double &lhs_value, const double &rhs_value) -> Number {
				return Number{ std::pow(lhs_value, rhs_value) };
			},
		},
		value,
		rhs.value);
}

Number Number::operator+(const Number &rhs) const
{
	return std::visit(overloaded{
						  [](const mpz_class &lhs_value, const mpz_class &rhs_value) -> Number {
							  return Number{ lhs_value + rhs_value };
						  },
						  [](const mpz_class &lhs_value, const double &rhs_value) -> Number {
							  return Number{ lhs_value.get_d() + rhs_value };
						  },
						  [](const double &lhs_value, const mpz_class &rhs_value) -> Number {
							  return Number{ lhs_value + rhs_value.get_d() };
						  },
						  [](const double &lhs_value, const double &rhs_value) -> Number {
							  return Number{ lhs_value + rhs_value };
						  },
					  },
		value,
		rhs.value);
}

Number Number::operator-(const Number &rhs) const
{
	return std::visit(overloaded{
						  [](const mpz_class &lhs_value, const mpz_class &rhs_value) -> Number {
							  return Number{ lhs_value - rhs_value };
						  },
						  [](const mpz_class &lhs_value, const double &rhs_value) -> Number {
							  return Number{ lhs_value.get_d() - rhs_value };
						  },
						  [](const double &lhs_value, const mpz_class &rhs_value) -> Number {
							  return Number{ lhs_value - rhs_value.get_d() };
						  },
						  [](const double &lhs_value, const double &rhs_value) -> Number {
							  return Number{ lhs_value - rhs_value };
						  },
					  },
		value,
		rhs.value);
}

Number Number::operator%(const Number &rhs) const
{
	return std::visit(overloaded{
						  [](const mpz_class &lhs_value, const mpz_class &rhs_value) -> Number {
							  return Number{ lhs_value % rhs_value };
						  },
						  [](const mpz_class &lhs_value, const double &rhs_value) -> Number {
							  return Number{ std::fmod(lhs_value.get_d(), rhs_value) };
						  },
						  [](const double &lhs_value, const mpz_class &rhs_value) -> Number {
							  return Number{ std::fmod(lhs_value, rhs_value.get_d()) };
						  },
						  [](const double &lhs_value, const double &rhs_value) -> Number {
							  return Number{ std::fmod(lhs_value, rhs_value) };
						  },
					  },
		value,
		rhs.value);
}

Number Number::operator/(const Number &other) const
{
	return std::visit(overloaded{
						  [](const mpz_class &lhs_value, const mpz_class &rhs_value) -> Number {
							  return Number{ lhs_value.get_d() / rhs_value.get_d() };
						  },
						  [](const mpz_class &lhs_value, const double &rhs_value) -> Number {
							  return Number{ lhs_value.get_d() / rhs_value };
						  },
						  [](const double &lhs_value, const mpz_class &rhs_value) -> Number {
							  return Number{ lhs_value / rhs_value.get_d() };
						  },
						  [](const double &lhs_value, const double &rhs_value) -> Number {
							  return Number{ lhs_value / rhs_value };
						  },
					  },
		value,
		other.value);
}

Number Number::operator*(const Number &other) const
{
	return std::visit(overloaded{
						  [](const mpz_class &lhs_value, const mpz_class &rhs_value) -> Number {
							  return Number{ lhs_value * rhs_value };
						  },
						  [](const mpz_class &lhs_value, const double &rhs_value) -> Number {
							  return Number{ lhs_value.get_d() * rhs_value };
						  },
						  [](const double &lhs_value, const mpz_class &rhs_value) -> Number {
							  return Number{ lhs_value * rhs_value.get_d() };
						  },
						  [](const double &lhs_value, const double &rhs_value) -> Number {
							  return Number{ lhs_value * rhs_value };
						  },
					  },
		value,
		other.value);
}

bool Number::operator==(const PyObject *other) const
{
	if (auto other_pynumber = as<PyInteger>(other)) {
		return *this == other_pynumber->value();
	} else if (auto other_pynumber = as<PyFloat>(other)) {
		return *this == other_pynumber->value();
	} else {
		return false;
	}
}

bool Number::operator==(const NameConstant &other) const
{
	if (std::holds_alternative<NoneType>(other.value)) { return false; }
	if (*this == Number{ int64_t{ 0 } }) { return std::get<bool>(other.value) == false; }
	if (*this == Number{ int64_t{ 1 } }) { return std::get<bool>(other.value) == true; }
	return false;
}

bool Number::operator==(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) { return lhs_value == rhs_value; },
		value,
		rhs.value);
}

Number Number::operator<<(const Number &rhs) const
{
	return std::visit(overloaded{
						  [](const mpz_class &lhs_value, const mpz_class &rhs_value) -> Number {
							  ASSERT(rhs_value.fits_ulong_p());
							  return Number{ lhs_value << rhs_value.get_ui() };
						  },
						  [](const mpz_class &, const double &) -> Number {
							  // should raise error
							  TODO();
						  },
						  [](const double &, const mpz_class &) -> Number {
							  // should raise error
							  TODO();
						  },
						  [](const double &, const double &) -> Number {
							  // should raise error
							  TODO();
						  },
					  },
		value,
		rhs.value);
}

Number Number::operator>>(const Number &rhs) const
{
	return std::visit(overloaded{
						  [](const BigIntType &lhs_value, const BigIntType &rhs_value) -> Number {
							  ASSERT(rhs_value.fits_ulong_p());
							  return Number{ lhs_value >> rhs_value.get_ui() };
						  },
						  [](const BigIntType &, const double &) -> Number {
							  // should raise error
							  TODO();
						  },
						  [](const double &, const BigIntType &) -> Number {
							  // should raise error
							  TODO();
						  },
						  [](const double &, const double &) -> Number {
							  // should raise error
							  TODO();
						  },
					  },
		value,
		rhs.value);
}

bool Number::operator<=(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) -> bool { return lhs_value <= rhs_value; },
		value,
		rhs.value);
}
bool Number::operator<(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) -> bool { return lhs_value < rhs_value; },
		value,
		rhs.value);
}

bool Number::operator>(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) -> bool { return lhs_value > rhs_value; },
		value,
		rhs.value);
}

bool Number::operator>=(const Number &rhs) const
{
	return std::visit(
		[](const auto &lhs_value, const auto &rhs_value) -> bool { return lhs_value >= rhs_value; },
		value,
		rhs.value);
}

Number Number::floordiv(const Number &other) const
{
	return std::visit(
		overloaded{
			[](const double &lhs, const BigIntType &rhs) -> Number {
				return Number{ BigIntType{ BigIntType{ lhs } / rhs }.get_d() };
			},
			[](const BigIntType &lhs, const double &rhs) -> Number {
				return Number{ BigIntType{ lhs / BigIntType{ rhs } }.get_d() };
			},
			[](const BigIntType &lhs, const BigIntType &rhs) -> Number {
				return Number{ BigIntType{ lhs / rhs } };
			},
			[](const double &lhs, const double &rhs) -> Number {
				return Number{ BigIntType{ BigIntType{ lhs } / BigIntType{ rhs } }.get_d() };
			},
		},
		value,
		other.value);
}

String String::from_unescaped_string(const std::string &str)
{
	std::string output;
	auto it = str.begin();
	const auto end = str.end();
	while (it != end) {
		if (auto c = *it++; c != '\\') {
			output.push_back(static_cast<unsigned char>(c));
			continue;
		}

		if (it == end) {
			// return Err(value_error("Trailing \\ in string"));
			TODO();
		}
		switch (*it++) {
		case '\n':
			break;
		case '\\': {
			output.push_back('\\');
		} break;
		case '\'': {
			output.push_back('\'');
		} break;
		case '\"': {
			output.push_back('\"');
		} break;
		case 'b': {
			output.push_back('\b');
		} break;
		case 'f': {
			output.push_back('\014');
		} break;
		case 't': {
			output.push_back('\t');
		} break;
		case 'n': {
			output.push_back('\n');
		} break;
		case 'r': {
			output.push_back('\r');
		} break;
		case 'v': {
			output.push_back('\013');
		} break;
		case 'a': {
			output.push_back('\007');
		} break;
		case '0':
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7': {
			auto c = *(it - 1) - '0';
			if (it != end && '0' <= *it && *it <= '7') {
				c = (c << 3) + *it++ - '0';
				if (it != end && '0' <= *it && *it <= '7') { c = (c << 3) + *it++ - '0'; }
			}
			ASSERT(c < static_cast<int>(std::numeric_limits<unsigned char>::max()));
			output.push_back(static_cast<unsigned char>(c));
		} break;
		case 'x': {
			TODO();
		} break;
		case 'u': {
			TODO();
		} break;
		case 'U': {
			TODO();
		} break;
		case 'N': {
			TODO();
		} break;
		default: {
			output.push_back('\\');
			output.push_back(static_cast<unsigned char>(*(it - 1)));
		} break;
		}
	}

	return String{ std::move(output) };
}

bool String::operator==(const PyObject *other) const
{
	if (auto other_pystr = as<PyString>(other)) {
		return s == other_pystr->value();
	} else {
		return false;
	}
}


bool Bytes::operator==(const PyObject *other) const
{
	if (auto other_pybytes = as<PyBytes>(other)) {
		return *this == other_pybytes->value();
	} else {
		return false;
	}
}

Bytes Bytes::from_unescaped_string(const std::string &str)
{
	std::vector<std::byte> bytes;
	auto it = str.begin();
	const auto end = str.end();
	while (it != end) {
		if (auto c = *it++; c != '\\') {
			bytes.push_back(std::byte{ static_cast<unsigned char>(c) });
			continue;
		}

		if (it == end) {
			// return Err(value_error("Trailing \\ in string"));
			TODO();
		}

		switch (*it++) {
		case '\n':
			break;
		case '\\': {
			bytes.emplace_back(std::byte{ '\\' });
		} break;
		case '\'': {
			bytes.emplace_back(std::byte{ '\'' });
		} break;
		case '\"': {
			bytes.emplace_back(std::byte{ '\"' });
		} break;
		case 'b': {
			bytes.emplace_back(std::byte{ '\b' });
		} break;
		case 'f': {
			bytes.emplace_back(std::byte{ '\014' });
		} break;
		case 't': {
			bytes.emplace_back(std::byte{ '\t' });
		} break;
		case 'n': {
			bytes.emplace_back(std::byte{ '\n' });
		} break;
		case 'r': {
			bytes.emplace_back(std::byte{ '\r' });
		} break;
		case 'v': {
			bytes.emplace_back(std::byte{ '\013' });
		} break;
		case 'a': {
			bytes.emplace_back(std::byte{ '\007' });
		} break;
		case '0':
		case '1':
		case '2':
		case '3':
		case '4':
		case '5':
		case '6':
		case '7': {
			auto c = *(it - 1) - '0';
			if (it != end && '0' <= *it && *it <= '7') {
				c = (c << 3) + *it++ - '0';
				if (it != end && '0' <= *it && *it <= '7') { c = (c << 3) + *it++ - '0'; }
			}
			ASSERT(c < static_cast<int>(std::numeric_limits<unsigned char>::max()));
			bytes.push_back(std::byte{static_cast<unsigned char>(c)});
		} break;
		case 'x': {
			TODO();
		} break;
		default: {
			bytes.push_back(std::byte{'\\'});
			--it;
		} break;
		}
	}

	return Bytes{ std::move(bytes) };
}


std::string Bytes::to_string() const
{
	static constexpr std::string_view hex_digits = "0123456789abcdef";
	std::ostringstream os;
	os << "b'";
	for (const auto &byte : b) {
		const auto byte_ = std::to_integer<unsigned char>(byte);
		if (byte_ == '\\') {
			os << "\\";
		} else if (byte_ == '\t') {
			os << "\\t";
		} else if (byte_ == '\n') {
			os << "\\n";
		} else if (byte_ == '\r') {
			os << "\\r";
		} else if (byte_ < ' ' || byte_ >= 0x7f) {
			os << "\\x" << hex_digits[(byte_ & 0xf0) >> 4] << hex_digits[byte_ & 0xf];
		} else {
			os << byte_;
		}
	}
	os << "'";
	return os.str();
}

bool Ellipsis::operator==(const PyObject *other) const { return other == py_ellipsis(); }


bool NoneType::operator==(const PyObject *other) const { return other == py_none(); }


bool NameConstant::operator==(const PyObject *other) const
{
	if (std::holds_alternative<NoneType>(value)) { return other == py_none(); }
	const auto bool_value = std::get<bool>(value);
	if (bool_value) {
		return other == py_true();
	} else {
		return other == py_false();
	}
}


bool NameConstant::operator==(const Number &other) const
{
	if (std::holds_alternative<NoneType>(value)) { return false; }
	const short bool_value = std::get<bool>(value);
	if (bool_value) {
		return other == Number{ int64_t{ 1 } };
	} else {
		return other == Number{ int64_t{ 0 } };
	}
}

bool NameConstant::operator==(const NameConstant &other) const
{
	return std::visit(
		[](const auto &rhs, const auto &lhs) { return rhs == lhs; }, value, other.value);
}

namespace py {

PyResult<Value> add(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value + rhs_value);
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->add(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> subtract(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value - rhs_value);
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->subtract(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> multiply(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value * rhs_value);
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->multiply(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> exp(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value.exp(rhs_value));
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->exp(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> lshift(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value << rhs_value);
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->lshift(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> rshift(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value >> rhs_value);
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->rshift(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> modulo(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value % rhs_value);
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->modulo(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> true_divide(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value / rhs_value);
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) { return py_lhs; }
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) { return py_rhs; }
				return py_lhs.unwrap()->truediv(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> floordiv(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
					   return Ok(lhs_value.floordiv(rhs_value));
				   },
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) { return py_lhs; }
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) { return py_rhs; }
				return py_lhs.unwrap()->floordiv(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> equals(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult<Value> {
					   return Ok(NameConstant{ lhs_value == rhs_value });
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value == rhs_value });
			},
			[](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value == rhs_value });
			},
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->richcompare(py_rhs.unwrap(), RichCompare::Py_EQ);
			} },
		lhs,
		rhs);
}

PyResult<Value> not_equals(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult<Value> {
					   return Ok(NameConstant{ lhs_value != rhs_value });
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value != rhs_value });
			},
			[](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value != rhs_value });
			},
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->richcompare(py_rhs.unwrap(), RichCompare::Py_NE);
			} },
		lhs,
		rhs);
}

PyResult<Value> less_than_equals(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult<Value> {
					   return Ok(NameConstant{ lhs_value <= rhs_value });
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value <= rhs_value });
			},
			[](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value <= rhs_value });
			},
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->richcompare(py_rhs.unwrap(), RichCompare::Py_LE);
			} },
		lhs,
		rhs);
}


PyResult<Value> less_than(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult<Value> {
					   return Ok(NameConstant{ lhs_value < rhs_value });
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value < rhs_value });
			},
			[](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value < rhs_value });
			},
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->richcompare(py_rhs.unwrap(), RichCompare::Py_LT);
			} },
		lhs,
		rhs);
}

PyResult<Value> greater_than(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult<Value> {
					   return Ok(NameConstant{ lhs_value > rhs_value });
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value > rhs_value });
			},
			[](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value > rhs_value });
			},
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->richcompare(py_rhs.unwrap(), RichCompare::Py_GT);
			} },
		lhs,
		rhs);
}

PyResult<Value> greater_than_equals(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{ [](const NoneType &lhs_value, const auto &rhs_value) -> PyResult<Value> {
					   return Ok(NameConstant{ lhs_value >= rhs_value });
				   },
			[](const auto &lhs_value, const NoneType &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value >= rhs_value });
			},
			[](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
				return Ok(NameConstant{ lhs_value >= rhs_value });
			},
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->richcompare(py_rhs.unwrap(), RichCompare::Py_GE);
			} },
		lhs,
		rhs);
}

PyResult<Value> and_(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{
			[](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
				if (std::holds_alternative<BigIntType>(lhs_value.value)
					&& std::holds_alternative<BigIntType>(rhs_value.value)) {
					return Ok(Number{ std::get<BigIntType>(lhs_value.value)
									  & std::get<BigIntType>(rhs_value.value) });
				} else {
					const std::string lhs_type =
						std::holds_alternative<BigIntType>(lhs_value.value) ? "int" : "float";
					const std::string rhs_type =
						std::holds_alternative<BigIntType>(rhs_value.value) ? "int" : "float";
					return Err(type_error(
						"unsupported operand type(s) for &: '{}' and '{}'", lhs_type, rhs_type));
				}
			},
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->and_(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<Value> or_(const Value &lhs, const Value &rhs, Interpreter &)
{
	return std::visit(
		overloaded{
			[](const Number &lhs_value, const Number &rhs_value) -> PyResult<Value> {
				if (std::holds_alternative<BigIntType>(lhs_value.value)
					&& std::holds_alternative<BigIntType>(rhs_value.value)) {
					return Ok(Number{ std::get<BigIntType>(lhs_value.value)
									  | std::get<BigIntType>(rhs_value.value) });
				} else {
					const std::string lhs_type =
						std::holds_alternative<BigIntType>(lhs_value.value) ? "int" : "float";
					const std::string rhs_type =
						std::holds_alternative<BigIntType>(rhs_value.value) ? "int" : "float";
					return Err(type_error(
						"unsupported operand type(s) for &: '{}' and '{}'", lhs_type, rhs_type));
				}
			},
			[](const auto &lhs_value, const auto &rhs_value) -> PyResult<Value> {
				const auto py_lhs = PyObject::from(lhs_value);
				if (py_lhs.is_err()) return py_lhs;
				const auto py_rhs = PyObject::from(rhs_value);
				if (py_rhs.is_err()) return py_rhs;
				return py_lhs.unwrap()->or_(py_rhs.unwrap());
			} },
		lhs,
		rhs);
}

PyResult<bool> is(const Value &lhs, const Value &rhs, Interpreter &)
{
	// TODO: Could probably be more efficient, but at least guarantees that Python singletons
	//		 always are true in this comparisson when compared to the same singleton
	auto lhs_ = PyObject::from(lhs);
	if (lhs_.is_err()) return lhs_;
	auto rhs_ = PyObject::from(rhs);
	if (rhs_.is_err()) return rhs_;

	return Ok(lhs_.unwrap() == rhs_.unwrap());
}

PyResult<bool> in(const Value &lhs, const Value &rhs, Interpreter &)
{
	if (!std::holds_alternative<PyObject *>(rhs)) {
		return Err(type_error("argument of type '{}' is not iterable",
			PyObject::from(rhs).unwrap()->type()->to_string()));
	}
	auto value = PyObject::from(lhs);
	if (value.is_err()) { return Err(value.unwrap_err()); }
	return std::get<PyObject *>(rhs)->contains(value.unwrap());
}

PyResult<bool> truthy(const Value &value, Interpreter &)
{
	// Number, String, Bytes, Ellipsis, NameConstant, PyObject *
	return std::visit(overloaded{ [](const NameConstant &c) -> PyResult<bool> {
									 if (std::holds_alternative<NoneType>(c.value)) {
										 return Ok(false);
									 } else {
										 return Ok(std::get<bool>(c.value));
									 }
								 },
						  [](const Number &number) -> PyResult<bool> {
							  return Ok(number != Number{ int64_t{ 0 } });
						  },
						  [](const String &str) -> PyResult<bool> { return Ok(!str.s.empty()); },
						  [](const Bytes &bytes) -> PyResult<bool> { return Ok(!bytes.b.empty()); },
						  [](const Ellipsis &) -> PyResult<bool> { return Ok(true); },
						  [](PyObject *obj) -> PyResult<bool> { return obj->true_(); } },
		value);
}

bool operator==(const Value &lhs_value, const Value &rhs_value)
{
	const auto result =
		std::visit(overloaded{ [](PyObject *const lhs, PyObject *const rhs) {
								  auto r = lhs->richcompare(rhs, RichCompare::Py_EQ);
								  ASSERT(r.is_ok())
								  return r.unwrap() == py_true();
							  },
					   [](PyObject *const lhs, const auto &rhs) { return lhs == rhs; },
					   [](const auto &lhs, PyObject *const rhs) { return lhs == rhs; },
					   [](const auto &lhs, const auto &rhs) { return lhs == rhs; } },
			lhs_value,
			rhs_value);
	return result;
}

}// namespace py
