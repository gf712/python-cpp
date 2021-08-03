#include "PyObject.hpp"


class PyNumber final : public PyObject
{
	friend class Heap;

	Number m_value;

  public:
	static std::shared_ptr<PyNumber> create(const Number &number);
	std::string to_string() const override
	{
		return std::visit(
			[](const auto &value) { return fmt::format("{}", value); }, m_value.value);
	}

	std::shared_ptr<PyObject> add_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const override;
	std::shared_ptr<PyObject> subtract_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const override;
	std::shared_ptr<PyObject> modulo_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const override;
	std::shared_ptr<PyObject> multiply_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const override;

	std::shared_ptr<PyObject> repr_impl(Interpreter &interpreter) const override;
	virtual std::shared_ptr<PyObject> equal_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const override;

	const Number &value() const { return m_value; }

  private:
	PyNumber(Number number) : PyObject(PyObjectType::PY_NUMBER), m_value(number) {}
};


template<> inline std::shared_ptr<PyNumber> as(std::shared_ptr<PyObject> node)
{
	if (node->type() == PyObjectType::PY_NUMBER) {
		return std::static_pointer_cast<PyNumber>(node);
	}
	return nullptr;
}