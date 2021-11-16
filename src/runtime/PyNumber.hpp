#include "PyObject.hpp"


class PyNumber final : public PyBaseObject<PyNumber>
{
	friend class Heap;

	Number m_value;

  public:
	static PyNumber *create(const Number &number);
	std::string to_string() const override
	{
		return std::visit(
			[](const auto &value) { return fmt::format("{}", value); }, m_value.value);
	}

	PyObject *add_impl(const PyObject *obj) const;
	PyObject *subtract_impl(const PyObject *obj, Interpreter &interpreter) const override;
	PyObject *modulo_impl(const PyObject *obj, Interpreter &interpreter) const override;
	PyObject *multiply_impl(const PyObject *obj, Interpreter &interpreter) const override;

	PyObject *repr_impl() const;
	PyObject *equal_impl(const PyObject *obj, Interpreter &interpreter) const override;
	PyObject *less_than_impl(const PyObject *obj, Interpreter &interpreter) const override;
	PyObject *less_than_equal_impl(const PyObject *obj, Interpreter &interpreter) const override;
	PyObject *greater_than_impl(const PyObject *obj, Interpreter &interpreter) const override;
	PyObject *greater_than_equal_impl(const PyObject *obj, Interpreter &interpreter) const override;

	const Number &value() const { return m_value; }

  private:
	PyNumber(Number number) : PyBaseObject(PyObjectType::PY_NUMBER), m_value(number) {}
};


template<> inline PyNumber *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_NUMBER) { return static_cast<PyNumber *>(node); }
	return nullptr;
}

template<> inline const PyNumber *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_NUMBER) { return static_cast<const PyNumber *>(node); }
	return nullptr;
}
