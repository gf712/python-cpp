#pragma once

#include "Value.hpp"
#include "utilities.hpp"

#include <unordered_map>
#include <string>
#include <memory>
#include <functional>
#include <sstream>

#include <spdlog/fmt/fmt.h>


enum class PyObjectType {
	PY_STRING,
	PY_NUMBER,
	PY_FUNCTION,
	PY_NATIVE_FUNCTION,
	PY_BYTES,
	PY_ELLIPSIS,
	PY_CONSTANT_NAME,
	PY_CODE,
	PY_LIST,
	PY_LIST_ITERATOR,
	PY_BASE_EXCEPTION,
	PY_RANGE,
	PY_RANGE_ITERATOR
};

inline std::string_view object_name(PyObjectType type)
{
	switch (type) {
	case PyObjectType::PY_STRING: {
		return "string";
	}
	case PyObjectType::PY_NUMBER: {
		return "number";
	}
	case PyObjectType::PY_FUNCTION: {
		return "function";
	}
	case PyObjectType::PY_BYTES: {
		return "bytes";
	}
	case PyObjectType::PY_ELLIPSIS: {
		return "ellipsis";
	}
	case PyObjectType::PY_CONSTANT_NAME: {
		return "constant_name";
	}
	case PyObjectType::PY_NATIVE_FUNCTION: {
		return "external_function";
	}
	case PyObjectType::PY_CODE: {
		return "code";
	}
	case PyObjectType::PY_LIST: {
		return "list";
	}
	case PyObjectType::PY_LIST_ITERATOR: {
		return "list_iterator";
	}
	case PyObjectType::PY_BASE_EXCEPTION: {
		return "BaseException";
	}
	case PyObjectType::PY_RANGE: {
		return "range";
	}
	case PyObjectType::PY_RANGE_ITERATOR: {
		return "range_iterator";
	}
	}
	ASSERT_NOT_REACHED()
}

class PyObject : public std::enable_shared_from_this<PyObject>
{
	const PyObjectType m_type;

  protected:
  public:
	PyObject(PyObjectType type);

	virtual ~PyObject() = default;
	PyObjectType type() const { return m_type; }
	std::string_view type_string() const { return object_name(m_type); }

	virtual std::string to_string() const = 0;

	virtual std::shared_ptr<PyObject> add_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const;
	virtual std::shared_ptr<PyObject> subtract_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const;
	virtual std::shared_ptr<PyObject> multiply_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const;
	virtual std::shared_ptr<PyObject> exp_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const;
	virtual std::shared_ptr<PyObject> lshift_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const;
	virtual std::shared_ptr<PyObject> repr_impl(Interpreter &interpreter) const;
	virtual std::shared_ptr<PyObject> equal_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const;
	virtual std::shared_ptr<PyObject> iter_impl(Interpreter &interpreter) const;
	virtual std::shared_ptr<PyObject> next_impl(Interpreter &interpreter);
	virtual std::shared_ptr<PyObject> len_impl(Interpreter &interpreter) const;

	template<typename T> static std::shared_ptr<PyObject> from(const T &value);

  protected:
	template<typename T> std::shared_ptr<const T> shared_from_this_as() const
	{
		return std::static_pointer_cast<const T>(shared_from_this());
	}
};


class PyCode : public PyObject
{
	size_t m_pos;
	size_t m_register_count;
	std::vector<std::string> m_args;

  public:
	PyCode(const size_t pos, const size_t register_count, std::vector<std::string> args);

	size_t offset() const { return m_pos; }
	size_t register_count() const { return m_register_count; }
	const std::vector<std::string> &args() const { return m_args; }

	std::string to_string() const override { return fmt::format("PyCode"); }
};


class PyFunction : public PyObject
{
	std::shared_ptr<PyCode> m_code;

  public:
	PyFunction(std::shared_ptr<PyCode> code);

	const std::shared_ptr<PyCode> &code() const { return m_code; }

	std::string to_string() const override { return fmt::format("PyFunction"); }
};


class PyNativeFunction : public PyObject
{
	std::function<std::shared_ptr<PyObject>(std::shared_ptr<PyObject>)> m_function;

  public:
	PyNativeFunction(
		std::function<std::shared_ptr<PyObject>(const std::shared_ptr<PyObject> &)> function)
		: PyObject(PyObjectType::PY_NATIVE_FUNCTION), m_function(function)
	{}

	std::shared_ptr<PyObject> operator()(const std::shared_ptr<PyObject> &args)
	{
		return m_function(args);
	}

	std::string to_string() const override
	{
		return fmt::format("PyNativeFunction {}", static_cast<const void *>(&m_function));
	}
};

class PyString : public PyObject
{
	friend class Heap;
	std::string m_value;

  public:
	static std::shared_ptr<PyString> create(const std::string &value);

	~PyString() override = default;
	const std::string &value() const { return m_value; }

	std::string to_string() const override { return fmt::format("{}", m_value); }

	std::shared_ptr<PyObject> add_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const override;
	std::shared_ptr<PyObject> repr_impl(Interpreter &interpreter) const override;


  private:
	PyString(std::string s) : PyObject(PyObjectType::PY_STRING), m_value(std::move(s)) {}
};

class PyObjectNumber final : public PyObject
{
	friend class Heap;

	Number m_value;

  public:
	static std::shared_ptr<PyObjectNumber> create(const Number &number);
	~PyObjectNumber() {}
	std::string to_string() const override
	{
		std::ostringstream os;
		os << m_value;
		return fmt::format("PyObjectNumber {}", os.str());
	}

	std::shared_ptr<PyObject> add_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const override;
	std::shared_ptr<PyObject> repr_impl(Interpreter &interpreter) const override;
	virtual std::shared_ptr<PyObject> equal_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const override;

	const Number &value() const { return m_value; }

  private:
	PyObjectNumber(Number number) : PyObject(PyObjectType::PY_NUMBER), m_value(number)
	{
		// m_attributes.emplace("__repr__",
		// 	std::make_shared<PyNativeFunction>([this](const std::shared_ptr<PyObject> &) {
		// 		return PyString::create(this->to_string());
		// 	}));
		// m_attributes.emplace(
		// 	"__add__", std::make_shared<PyFunction>([this](const std::shared_ptr<PyObject> &) {
		// 		return PyString::create(this->to_string());
		// 	}));
	}
};

class PyBytes : public PyObject
{
	friend class Heap;

	Bytes m_value;

  public:
	static std::shared_ptr<PyBytes> create(const Bytes &number);
	~PyBytes() = default;
	std::string to_string() const override
	{
		std::ostringstream os;
		os << m_value;
		return fmt::format("PyBytes {}", os.str());
	}

	std::shared_ptr<PyObject> add_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const override;

	const Bytes &value() const { return m_value; }

  private:
	PyBytes(const Bytes &number) : PyObject(PyObjectType::PY_BYTES), m_value(number) {}
};


class PyEllipsis : public PyObject
{
	friend class Heap;
	friend std::shared_ptr<PyObject> py_ellipsis();

	static constexpr Ellipsis m_value{};

  public:
	std::string to_string() const override { return fmt::format("PyEllipsis"); }

	std::shared_ptr<PyObject> add_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const override;

	const Ellipsis &value() const { return m_value; }

  private:
	static std::shared_ptr<PyEllipsis> create();
	PyEllipsis() : PyObject(PyObjectType::PY_ELLIPSIS) {}
};


class PyNameConstant : public PyObject
{
	friend class Heap;
	friend std::shared_ptr<PyObject> py_none();
	friend std::shared_ptr<PyObject> py_false();
	friend std::shared_ptr<PyObject> py_true();

	NameConstant m_value;

  public:
	std::string to_string() const override;

	std::shared_ptr<PyObject> add_impl(const std::shared_ptr<PyObject> &obj,
		Interpreter &interpreter) const override;

	std::shared_ptr<PyObject> repr_impl(Interpreter &interpreter) const override;

	const NameConstant &value() const { return m_value; }

  private:
	static std::shared_ptr<PyNameConstant> create(const NameConstant &);

	PyNameConstant(const NameConstant &name)
		: PyObject(PyObjectType::PY_CONSTANT_NAME), m_value(name)
	{}
};

class PyList : public PyObject
{
	friend class Heap;

	std::vector<Value> m_elements;

  public:
	PyList(std::vector<Value> elements)
		: PyObject(PyObjectType::PY_LIST), m_elements(std::move(elements))
	{}

	std::string to_string() const override;

	std::shared_ptr<PyObject> repr_impl(Interpreter &interpreter) const override;
	std::shared_ptr<PyObject> iter_impl(Interpreter &interpreter) const override;

	const std::vector<Value> &elements() const { return m_elements; }
};


class PyListIterator : public PyObject
{
	friend class Heap;

	std::shared_ptr<const PyList> m_pylist;
	size_t m_current_index;

  public:
	PyListIterator(std::shared_ptr<const PyList> pylist)
		: PyObject(PyObjectType::PY_LIST_ITERATOR), m_pylist(std::move(pylist))
	{}

	std::string to_string() const override;

	std::shared_ptr<PyObject> repr_impl(Interpreter &interpreter) const override;
	std::shared_ptr<PyObject> next_impl(Interpreter &interpreter) override;
};


template<typename T> std::shared_ptr<T> as(std::shared_ptr<PyObject> node);


template<> inline std::shared_ptr<PyFunction> as(std::shared_ptr<PyObject> node)
{
	if (node->type() == PyObjectType::PY_FUNCTION) {
		return std::static_pointer_cast<PyFunction>(node);
	}
	return nullptr;
}


template<> inline std::shared_ptr<PyString> as(std::shared_ptr<PyObject> node)
{
	if (node->type() == PyObjectType::PY_STRING) {
		return std::static_pointer_cast<PyString>(node);
	}
	return nullptr;
}

template<> inline std::shared_ptr<PyObjectNumber> as(std::shared_ptr<PyObject> node)
{
	if (node->type() == PyObjectType::PY_NUMBER) {
		return std::static_pointer_cast<PyObjectNumber>(node);
	}
	return nullptr;
}

template<> inline std::shared_ptr<PyNameConstant> as(std::shared_ptr<PyObject> node)
{
	if (node->type() == PyObjectType::PY_CONSTANT_NAME) {
		return std::static_pointer_cast<PyNameConstant>(node);
	}
	return nullptr;
}


template<> inline std::shared_ptr<PyNativeFunction> as(std::shared_ptr<PyObject> node)
{
	if (node->type() == PyObjectType::PY_NATIVE_FUNCTION) {
		return std::static_pointer_cast<PyNativeFunction>(node);
	}
	return nullptr;
}


template<> inline std::shared_ptr<PyList> as(std::shared_ptr<PyObject> node)
{
	if (node->type() == PyObjectType::PY_LIST) { return std::static_pointer_cast<PyList>(node); }
	return nullptr;
}

std::shared_ptr<PyObject> py_none();
std::shared_ptr<PyObject> py_true();
std::shared_ptr<PyObject> py_false();
std::shared_ptr<PyObject> py_ellipsis();
