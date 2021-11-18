#pragma once

#include "PyObject.hpp"

#include <optional>

class PyString : public PyBaseObject<PyString>
{
	friend class Heap;
	std::string m_value;

  public:
	static PyString *create(const std::string &value);

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

	PyString *capitalize() const;

	static void register_type(PyModule*);

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
