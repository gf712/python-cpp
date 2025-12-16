#include "Match.hpp"
#include "Pattern.hpp"
#include "runtime/PyArgParser.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyString.hpp"
#include "runtime/Value.hpp"
#include "runtime/modules/Modules.hpp"

#include <cctype>
#include <unicode/uchar.h>

#include <cstdint>
#include <limits>
#include <unicode/unistr.h>


namespace py {
PyResult<PyObject *> compile(PyTuple *args, PyDict *kwargs)
{
	auto result =
		PyArgsParser<PyObject *, int32_t, PyList *, size_t, PyDict *, PyTuple *>::unpack_tuple(args,
			kwargs,
			"_sre.compile",
			std::integral_constant<size_t, 6>{},
			std::integral_constant<size_t, 6>{});
	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [pattern, flags, code, groups, groupindex, indexgroup] = result.unwrap();

	auto obj = sre::Pattern::create(pattern, flags, code, groups, groupindex, indexgroup);

	return obj;
}

PyResult<PyObject *> unicode_iscased(PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<PyInteger *>::unpack_tuple(args,
		kwargs,
		"_sre.unicode_iscased",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 1>{});
	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [character] = result.unwrap();

	if (!character->as_big_int().fits_sint_p()) { return Ok(py_false()); }

	icu::UnicodeString original{ UChar32{ static_cast<int32_t>(character->as_i64()) } };
	auto other = original;
	other.toUpper();
	if (original != other) { return Ok(py_true()); }

	other = original;
	other.toLower();
	return Ok(other != original ? py_true() : py_false());
}

PyResult<PyObject *> unicode_tolower(PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<PyInteger *>::unpack_tuple(args,
		kwargs,
		"_sre.unicode_tolower",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 1>{});
	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [character] = result.unwrap();

	if (!character->as_big_int().fits_sint_p()) { return Ok(py_false()); }

	icu::UnicodeString original{ UChar32{ static_cast<int32_t>(character->as_i64()) } };
	original.toLower();

	return PyInteger::create(original.char32At(0));
}

PyResult<PyObject *> ascii_iscased(PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<PyInteger *>::unpack_tuple(args,
		kwargs,
		"_sre.ascii_iscased",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 1>{});
	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [character] = result.unwrap();

	if (!character->as_big_int().fits_sint_p()) { return Ok(py_false()); }

	auto ch = character->as_big_int().get_si();
	if (ch >= 128) { return Ok(py_false()); }
	return Ok(std::isalpha(static_cast<int>(ch)) ? py_true() : py_false());
}

PyResult<PyObject *> ascii_tolower(PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<PyInteger *>::unpack_tuple(args,
		kwargs,
		"_sre.ascii_tolower",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 1>{});
	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [character] = result.unwrap();

	auto ch = character->as_big_int().get_si();
	if (ch >= 128) { return Ok(character); }
	return PyInteger::create(std::tolower(static_cast<int>(ch)));
}


PyModule *sre_module()
{
	auto *module_ = PyModule::create(
		PyDict::create().unwrap(), PyString::create("_sre").unwrap(), PyString::create("").unwrap())
						.unwrap();

	module_->add_symbol(PyString::create("CODESIZE").unwrap(), Number{ BigIntType{ 4 } });

	module_->add_symbol(PyString::create("MAGIC").unwrap(), Number{ BigIntType{ 20171005 } });

	module_->add_symbol(PyString::create("MAXREPEAT").unwrap(),
		Number{ BigIntType{ std::numeric_limits<uint32_t>::max() } });

	module_->add_symbol(PyString::create("MAXGROUPS").unwrap(),
		Number{ BigIntType{ std::numeric_limits<uint32_t>::max() / 2 } });

	module_->add_symbol(PyString::create("compile").unwrap(),
		PyNativeFunction::create("compile", &compile).unwrap());

	module_->add_symbol(PyString::create("unicode_iscased").unwrap(),
		PyNativeFunction::create("unicode_iscased", &unicode_iscased).unwrap());

	module_->add_symbol(PyString::create("unicode_tolower").unwrap(),
		PyNativeFunction::create("unicode_tolower", &unicode_tolower).unwrap());

	module_->add_symbol(PyString::create("ascii_iscased").unwrap(),
		PyNativeFunction::create("ascii_iscased", &ascii_iscased).unwrap());

	module_->add_symbol(PyString::create("ascii_tolower").unwrap(),
		PyNativeFunction::create("ascii_tolower", &ascii_tolower).unwrap());

	(void)sre::Pattern::register_type(module_);
	(void)sre::Match::register_type(module_);

	return module_;
}
}// namespace py