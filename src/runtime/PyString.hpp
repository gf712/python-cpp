#pragma once

#include "PyObject.hpp"


class PyString : public PyObject
{
	friend class Heap;
	std::string m_value;

  public:
	static PyString *create(const std::string &value);

	const std::string &value() const { return m_value; }
	std::vector<int32_t> codepoints() const;
	std::optional<int32_t> codepoint() const;

	std::string to_string() const override { return fmt::format("{}", m_value); }

	PyObject *add_impl(const PyObject *obj, Interpreter &interpreter) const override;
	PyObject *repr_impl(Interpreter &interpreter) const override;
	size_t hash_impl(Interpreter &interpreter) const override;
	PyObject *equal_impl(const PyObject *obj, Interpreter &interpreter) const override;
	PyObject *less_than_impl(const PyObject *obj, Interpreter &) const override;

	PyObject *len_impl(Interpreter &interpreter) const override;

  private:
	PyString(std::string s);
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
