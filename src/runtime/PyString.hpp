#pragma once

#include "PyObject.hpp"


class PyString : public PyObject
{
	friend class Heap;
	std::string m_value;

  public:
	static std::shared_ptr<PyString> create(const std::string &value);

	const std::string &value() const { return m_value; }

	std::string to_string() const override { return fmt::format("{}", m_value); }

	std::shared_ptr<PyObject> add_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const override;
	std::shared_ptr<PyObject> repr_impl(Interpreter &interpreter) const override;
	size_t hash_impl(Interpreter &interpreter) const override;
	std::shared_ptr<PyObject> equal_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const override;
	std::shared_ptr<PyObject> richcompare_impl(const std::shared_ptr<PyObject> &,
		RichCompare,
		Interpreter &interpreter) const override;

  private:
	PyString(std::string s);
};


template<> inline std::shared_ptr<PyString> as(std::shared_ptr<PyObject> node)
{
	if (node->type() == PyObjectType::PY_STRING) {
		return std::static_pointer_cast<PyString>(node);
	}
	return nullptr;
}
