#pragma once

#include "PyObject.hpp"
#include "PyTuple.hpp"

#include <optional>

class PyString : public PyBaseObject<PyString>
{
	friend class Heap;
	std::string m_value;

  public:
	static PyString *create(const std::string &value);
	static PyString *create(PyString *self, PyTuple *args, PyDict *kwargs)
	{
		// FIXME with proper error handling
		ASSERT(self)
		ASSERT(!args || (args->size() == 0))
		ASSERT(!kwargs)

		return self;
	}

	const std::string &value() const { return m_value; }
	std::vector<int32_t> codepoints() const;
	std::optional<int32_t> codepoint() const;

	std::string to_string() const override { return fmt::format("{}", m_value); }

	PyObject *add_impl(const PyObject *obj) const;

	PyObject *repr_impl() const;
	size_t hash_impl(Interpreter &interpreter) const override;
	PyObject *equal_impl(const PyObject *obj, Interpreter &interpreter) const override;
	PyObject *less_than_impl(const PyObject *obj, Interpreter &) const override;

	PyObject *len_impl(Interpreter &interpreter) const override;

	PyObject *isalpha() const;
	PyObject *isalnum() const;
	PyObject *isascii() const;
	PyObject *isdigit() const;
	PyObject *islower() const;
	PyObject *isupper() const;

	PyString *capitalize() const;
	PyString *casefold() const;
	PyNumber *find(PyTuple *args, PyDict *kwargs) const;
	PyNumber *count(PyTuple *args, PyDict *kwargs) const;
	PyObject *endswith(PyTuple *args, PyDict *kwargs) const;
	PyString *join(PyTuple *args, PyDict *kwargs) const;
	PyString *lower() const;
	PyString *upper() const;

	static void register_type(PyModule *);

  private:
	PyString(std::string s);

	size_t get_position_from_slice(int64_t) const;
};


template<> inline PyString *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_STRING) { return static_cast<PyString *>(node); }
	return nullptr;
}


template<> inline const PyString *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_STRING) { return static_cast<const PyString *>(node); }
	return nullptr;
}
