#pragma once

#include "PyObject.hpp"
#include "PyTuple.hpp"

#include <optional>

namespace py {

class PyString : public PyBaseObject
{
	friend class ::Heap;
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

	std::string to_string() const override { return m_value; }

	static PyObject *__new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyObject *__repr__() const;
	size_t __hash__() const;
	PyObject *__eq__(const PyObject *obj) const;
	PyObject *__lt__(const PyObject *obj) const;

	PyObject *__len__() const;
	PyObject *__add__(const PyObject *obj) const;

	PyObject *isalpha() const;
	PyObject *isalnum() const;
	PyObject *isascii() const;
	PyObject *isdigit() const;
	PyObject *islower() const;
	PyObject *isupper() const;

	PyString *capitalize() const;
	PyString *casefold() const;
	PyInteger *find(PyTuple *args, PyDict *kwargs) const;
	PyInteger *count(PyTuple *args, PyDict *kwargs) const;
	PyObject *endswith(PyTuple *args, PyDict *kwargs) const;
	PyString *join(PyTuple *args, PyDict *kwargs) const;
	PyString *lower() const;
	PyString *upper() const;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;

  private:
	PyString(std::string s);

	size_t get_position_from_slice(int64_t) const;
};

}// namespace py