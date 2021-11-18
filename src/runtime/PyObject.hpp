#pragma once

#include "Value.hpp"
#include "concepts.hpp"
#include "forward.hpp"
#include "memory/GarbageCollector.hpp"
#include "runtime/forward.hpp"
#include "utilities.hpp"

#include <concepts>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>

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
	PY_TUPLE,
	PY_TUPLE_ITERATOR,
	PY_DICT,
	PY_DICT_ITEMS,
	PY_DICT_ITEMS_ITERATOR,
	PY_BASE_EXCEPTION,
	PY_RANGE,
	PY_RANGE_ITERATOR,
	PY_CUSTOM_TYPE,
	PY_MODULE,
	PY_TYPE,
	PY_METHOD_DESCRIPTOR,
	PY_SLOT_WRAPPER,
	PY_BOUND_METHOD
};

inline std::string_view object_name(PyObjectType type)
{
	switch (type) {
	case PyObjectType::PY_STRING: {
		return "str";
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
	case PyObjectType::PY_TUPLE: {
		return "tuple";
	}
	case PyObjectType::PY_TUPLE_ITERATOR: {
		return "tuple_iterator";
	}
	case PyObjectType::PY_DICT: {
		return "dict";
	}
	case PyObjectType::PY_DICT_ITEMS: {
		return "dict_items";
	}
	case PyObjectType::PY_DICT_ITEMS_ITERATOR: {
		return "dict_itemiterator";
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
	case PyObjectType::PY_CUSTOM_TYPE: {
		return "object";
	}
	case PyObjectType::PY_MODULE: {
		return "module";
	}
	case PyObjectType::PY_TYPE: {
		return "type";
	}
	case PyObjectType::PY_METHOD_DESCRIPTOR: {
		return "method_descriptor";
	}
	case PyObjectType::PY_SLOT_WRAPPER: {
		return "slot_wrapper";
	}
	case PyObjectType::PY_BOUND_METHOD: {
		return "bound method";
	}
	}
	ASSERT_NOT_REACHED()
}

enum class RichCompare {
	Py_LT = 0,// <
	Py_LE = 1,// <=
	Py_EQ = 2,// ==
	Py_NE = 3,// !=
	Py_GT = 4,// >
	Py_GE = 5,// >=
};


using NewSlotFunctionType = std::function<PyObject *(PyObject *, PyTuple *, PyDict *)>;
using InitSlotFunctionType = std::function<PyObject *(PyObject *, PyTuple *, PyDict *)>;


using ReprSlotFunctionType = std::function<PyObject *()>;
using IterSlotFunctionType = std::function<PyObject *()>;
using AddSlotFunctionType = std::function<PyObject *(PyObject *)>;
using HashSlotFunctionType = std::function<size_t()>;
using RichCompareSlotFunctionType = std::function<PyObject *(const PyObject *, RichCompare)>;

class PyObject;

class Slots
// : NonCopyable
// , NonMoveable
{
  public:
	std::variant<std::monostate, AddSlotFunctionType, PyFunction *> add;
	std::variant<std::monostate, NewSlotFunctionType, PyFunction *> new_;
	std::variant<ReprSlotFunctionType, PyFunction *> repr;
	std::variant<IterSlotFunctionType, PyFunction *> iter;
	std::variant<std::monostate, HashSlotFunctionType, PyFunction *> hash;
	std::variant<std::monostate, RichCompareSlotFunctionType, PyFunction *> richcompare;

  public:
	template<typename Derived>
	static Slots create(PyObject *object) requires std::is_base_of_v<PyObject, Derived>
	{
		return Slots{
			.add = put_add<Derived>(object),
			.repr = put_repr<Derived>(object),
		};
	}

  private:
	template<HasAdd T> static auto put_add(PyObject *) -> AddSlotFunctionType;
	template<typename T> static auto put_add(PyObject *) -> std::monostate;

	template<HasRepr T> static auto put_repr(PyObject *) -> ReprSlotFunctionType;
	// template<typename T> static auto put_repr(PyObject *);
};


class PyObject : public Cell
{
	const PyObjectType m_type;
	struct NotImplemented_
	{
	};

  protected:
	std::optional<Slots> m_slots;
	std::unordered_map<std::string, PyObject *> m_attributes;

  public:
	using PyResult = std::variant<PyObject *, NotImplemented_>;

	PyObject() = delete;
	PyObject(PyObjectType type) : Cell(), m_type(type) {}

	virtual ~PyObject() = default;
	PyObjectType type() const { return m_type; }
	std::string_view type_string() const { return object_name(m_type); }

	virtual PyObject *subtract_impl(const PyObject *obj, Interpreter &interpreter) const;
	virtual PyObject *multiply_impl(const PyObject *obj, Interpreter &interpreter) const;
	virtual PyObject *exp_impl(const PyObject *obj, Interpreter &interpreter) const;
	virtual PyObject *lshift_impl(const PyObject *obj, Interpreter &interpreter) const;
	virtual PyObject *modulo_impl(const PyObject *obj, Interpreter &interpreter) const;

	virtual PyObject *equal_impl(const PyObject *obj, Interpreter &interpreter) const;
	virtual PyObject *less_than_impl(const PyObject *obj, Interpreter &interpreter) const;
	virtual PyObject *less_than_equal_impl(const PyObject *obj, Interpreter &interpreter) const;
	virtual PyObject *greater_than_impl(const PyObject *obj, Interpreter &interpreter) const;
	virtual PyObject *greater_than_equal_impl(const PyObject *obj, Interpreter &interpreter) const;
	virtual PyObject *
		richcompare_impl(const PyObject *, RichCompare, Interpreter &interpreter) const;

	virtual PyObject *truthy(Interpreter &) const;

	virtual PyObject *iter_impl(Interpreter &interpreter) const;
	virtual PyObject *next_impl(Interpreter &interpreter);
	virtual PyObject *len_impl(Interpreter &interpreter) const;
	virtual size_t hash_impl(Interpreter &interpreter) const;

	template<typename T> static PyObject *from(const T &value);

	PyObject *get(std::string name, Interpreter &interpreter) const;
	void put(std::string name, PyObject *);
	const std::unordered_map<std::string, PyObject *> &attributes() const { return m_attributes; }

	PyObject *repr_impl() const;

	const Slots &slots() const
	{
		ASSERT(m_slots)
		return *m_slots;
	}

	void visit_graph(Visitor &) override;

	PyResult __add__(PyObject *other) const
	{
		ASSERT(m_slots)
		if (std::holds_alternative<std::monostate>(m_slots->add)) {
			return NotImplemented_{};
		} else if (std::holds_alternative<AddSlotFunctionType>(m_slots->add)) {
			return std::get<AddSlotFunctionType>(m_slots->add)(other);
		} else {
			TODO()
		}
	}

	PyObject *__repr__() const
	{
		ASSERT(m_slots)

		if (std::holds_alternative<ReprSlotFunctionType>(m_slots->repr)) {
			return std::get<ReprSlotFunctionType>(m_slots->repr)();
		} else {
			TODO()
		}
	}

	bool is_pyobject() const override { return true; }
};


template<HasAdd T> auto Slots::put_add(PyObject *obj) -> AddSlotFunctionType
{
	return [obj](PyObject *other) { return static_cast<T *>(obj)->add_impl(other); };
}

template<typename T> auto Slots::put_add(PyObject *) -> std::monostate { return std::monostate{}; }

template<HasRepr T> auto Slots::put_repr(PyObject *obj) -> ReprSlotFunctionType
{
	return [obj]() { return static_cast<T *>(obj)->repr_impl(); };
}

// template<typename T> auto Slots::put_repr(PyObject *)
// {
// 	[]<bool flag = false>() { static_assert(flag, "PyObject must implement repr function"); }
// 	();
// }

template<typename Derived> class PyBaseObject : public PyObject
{
  public:
	PyBaseObject(PyObjectType type) : PyObject(type) { m_slots = Slots::create<Derived>(this); }
};


class PyBytes : public PyBaseObject<PyBytes>
{
	friend class Heap;

	Bytes m_value;

  public:
	static PyBytes *create(const Bytes &number);
	~PyBytes() = default;
	std::string to_string() const override
	{
		std::ostringstream os;
		os << m_value;
		return fmt::format("PyBytes {}", os.str());
	}

	PyObject *add_impl(const PyObject *obj) const;

	const Bytes &value() const { return m_value; }

  private:
	PyBytes(const Bytes &number) : PyBaseObject(PyObjectType::PY_BYTES), m_value(number) {}
};


class PyEllipsis : public PyBaseObject<PyEllipsis>
{
	friend class Heap;
	friend PyObject *py_ellipsis();

	static constexpr Ellipsis m_value{};

  public:
	std::string to_string() const override { return fmt::format("PyEllipsis"); }

	PyObject *add_impl(const PyObject *obj) const;

	const Ellipsis &value() const { return m_value; }

  private:
	static PyEllipsis *create();
	PyEllipsis() : PyBaseObject(PyObjectType::PY_ELLIPSIS) {}
};


class PyNameConstant : public PyBaseObject<PyNameConstant>
{
	friend class Heap;
	friend class Env;

	NameConstant m_value;

  public:
	std::string to_string() const override;

	PyObject *add_impl(const PyObject *obj) const;

	PyObject *repr_impl() const;

	const NameConstant &value() const { return m_value; }

	void visit_graph(Visitor &) override {}

  private:
	static PyNameConstant *create(const NameConstant &);

	PyNameConstant(const NameConstant &name)
		: PyBaseObject(PyObjectType::PY_CONSTANT_NAME), m_value(name)
	{}
};


struct ValueHash
{
	size_t operator()(const Value &value) const;
};

struct ValueEqual
{
	bool operator()(const Value &lhs, const Value &rhs) const;
};


template<typename T> T *as(PyObject *node);
template<typename T> const T *as(const PyObject *node);


template<> inline PyNameConstant *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_CONSTANT_NAME) {
		return static_cast<PyNameConstant *>(node);
	}
	return nullptr;
}

template<> inline const PyNameConstant *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_CONSTANT_NAME) {
		return static_cast<const PyNameConstant *>(node);
	}
	return nullptr;
}


template<> inline PyBytes *as(PyObject *node)
{
	if (node->type() == PyObjectType::PY_BYTES) { return static_cast<PyBytes *>(node); }
	return nullptr;
}


template<> inline const PyBytes *as(const PyObject *node)
{
	if (node->type() == PyObjectType::PY_BYTES) { return static_cast<const PyBytes *>(node); }
	return nullptr;
}

PyObject *py_none();
PyObject *py_true();
PyObject *py_false();
PyObject *py_ellipsis();
