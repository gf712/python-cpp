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

enum class RichCompare {
	Py_LT = 0,// <
	Py_LE = 1,// <=
	Py_EQ = 2,// ==
	Py_NE = 3,// !=
	Py_GT = 4,// >
	Py_GE = 5,// >=
};

class PyObject;

struct MethodDefinition
{
	std::string name;
	std::function<PyObject *(PyObject * /* self */, PyTuple * /* args */, PyDict * /* kwargs */)>
		method;
};

using CallSlotFunctionType = std::function<PyObject *(PyObject *, PyTuple *, PyDict *)>;
using NewSlotFunctionType = std::function<PyObject *(const PyType *, PyTuple *, PyDict *)>;
using InitSlotFunctionType = std::function<std::optional<int32_t>(PyObject *, PyTuple *, PyDict *)>;
using LenSlotFunctionType = std::function<PyObject *(const PyObject *)>;
using BoolSlotFunctionType = std::function<PyObject *(const PyObject *)>;
using ReprSlotFunctionType = std::function<PyObject *(const PyObject *)>;
using IterSlotFunctionType = std::function<PyObject *(const PyObject *)>;
using NextSlotFunctionType = std::function<PyObject *(PyObject *)>;

using AbsSlotFunctionType = std::function<PyObject *(const PyObject *)>;
using NegSlotFunctionType = std::function<PyObject *(const PyObject *)>;
using PosSlotFunctionType = std::function<PyObject *(const PyObject *)>;
using InvertSlotFunctionType = std::function<PyObject *(const PyObject *)>;

using AddSlotFunctionType = std::function<PyObject *(const PyObject *, const PyObject *)>;
using SubtractSlotFunctionType = std::function<PyObject *(const PyObject *, const PyObject *)>;
using MultiplySlotFunctionType = std::function<PyObject *(const PyObject *, const PyObject *)>;
using ExpSlotFunctionType = std::function<PyObject *(const PyObject *, const PyObject *)>;
using LeftShiftSlotFunctionType = std::function<PyObject *(const PyObject *, const PyObject *)>;
using ModuloSlotFunctionType = std::function<PyObject *(const PyObject *, const PyObject *)>;

using HashSlotFunctionType = std::function<size_t(const PyObject *)>;
using CompareSlotFunctionType = std::function<PyObject *(const PyObject *, const PyObject *)>;

struct TypePrototype
{
	std::string __name__;
	std::optional<NewSlotFunctionType> __new__;
	std::optional<InitSlotFunctionType> __init__;
	PyType *__class__{ nullptr };

	std::optional<AddSlotFunctionType> __add__;
	std::optional<SubtractSlotFunctionType> __sub__;
	std::optional<MultiplySlotFunctionType> __mul__;
	std::optional<ExpSlotFunctionType> __exp__;
	std::optional<LeftShiftSlotFunctionType> __lshift__;
	std::optional<ModuloSlotFunctionType> __mod__;

	std::optional<AbsSlotFunctionType> __abs__;
	std::optional<AbsSlotFunctionType> __neg__;
	std::optional<AbsSlotFunctionType> __pos__;
	std::optional<AbsSlotFunctionType> __invert__;

	std::optional<CallSlotFunctionType> __call__;
	// std::variant<std::monostate, NewSlotFunctionType, PyFunction *> __new__;
	std::optional<LenSlotFunctionType> __len__;
	std::optional<BoolSlotFunctionType> __bool__;
	std::optional<ReprSlotFunctionType> __repr__;
	std::optional<IterSlotFunctionType> __iter__;
	std::optional<NextSlotFunctionType> __next__;
	std::optional<HashSlotFunctionType> __hash__;

	std::optional<CompareSlotFunctionType> __eq__;
	std::optional<CompareSlotFunctionType> __gt__;
	std::optional<CompareSlotFunctionType> __ge__;
	std::optional<CompareSlotFunctionType> __le__;
	std::optional<CompareSlotFunctionType> __lt__;
	std::optional<CompareSlotFunctionType> __ne__;
	// std::variant<std::monostate, RichCompareSlotFunctionType, PyFunction *> __richcompare__;
	std::vector<MethodDefinition> __methods__;
	PyDict *__dict__{ nullptr };

	PyTuple *__mro__{ nullptr };
	PyTuple *__bases__{ nullptr };

	template<typename Type> static std::unique_ptr<TypePrototype> create(std::string_view name);

	void add_method(MethodDefinition &&method) { __methods__.push_back(std::move(method)); }
};

class PyObject : public Cell
{
	struct NotImplemented_
	{
	};

	friend class Heap;

  protected:
	const TypePrototype &m_type_prototype;
	std::unordered_map<std::string, PyObject *> m_attributes;

  public:
	using PyResult = std::variant<PyObject *, NotImplemented_>;

	PyObject() = delete;
	PyObject(const TypePrototype &type);

	virtual ~PyObject() = default;

	virtual PyType *type() const = 0;

	template<typename T> static PyObject *from(const T &value);

	PyObject *get(std::string name, Interpreter &interpreter) const;
	void put(std::string name, PyObject *);
	const std::unordered_map<std::string, PyObject *> &attributes() const { return m_attributes; }

	void visit_graph(Visitor &) override;

	PyObject *add(const PyObject *other) const;
	PyObject *subtract(const PyObject *other) const;
	PyObject *multiply(const PyObject *other) const;
	PyObject *exp(const PyObject *other) const;
	PyObject *lshift(const PyObject *other) const;
	PyObject *modulo(const PyObject *other) const;

	PyObject *neg() const;
	PyObject *pos() const;
	PyObject *abs() const;
	PyObject *invert() const;

	PyObject *repr() const;

	size_t hash() const;

	PyObject *richcompare(const PyObject *other, RichCompare) const;
	PyResult eq(const PyObject *other) const;
	PyResult ge(const PyObject *other) const;
	PyResult gt(const PyObject *other) const;
	PyResult le(const PyObject *other) const;
	PyResult lt(const PyObject *other) const;
	PyResult ne(const PyObject *other) const;

	PyObject *bool_() const;
	PyObject *len() const;
	PyObject *iter() const;
	PyObject *next();

	PyObject *call(PyTuple *args, PyDict *kwargs);
	virtual PyObject *new_(PyTuple *args, PyDict *kwargs) const;
	std::optional<int32_t> init(PyTuple *args, PyDict *kwargs);

	PyObject *__eq__(const PyObject *other) const;
	PyObject *__repr__() const;
	size_t __hash__() const;
	PyObject *__bool__() const;

	bool is_pyobject() const override { return true; }
	bool is_callable() const;
	const std::string &name() const;
	const TypePrototype &type_prototype() const { return m_type_prototype; }
};

template<typename Type> std::unique_ptr<TypePrototype> TypePrototype::create(std::string_view name)
{
	auto type_prototype = std::make_unique<TypePrototype>();
	type_prototype->__name__ = std::string(name);
	if constexpr (HasRepr<Type>) {
		type_prototype->__repr__ =
			+[](const PyObject *self) { return static_cast<const Type *>(self)->__repr__(); };
	}
	if constexpr (HasCall<Type>) {
		type_prototype->__call__ = +[](PyObject *self, PyTuple *args, PyDict *kwargs) {
			return static_cast<Type *>(self)->__call__(args, kwargs);
		};
	}
	if constexpr (HasNew<Type>) {
		type_prototype->__new__ = +[](const PyType *type, PyTuple *args, PyDict *kwargs) {
			return Type::__new__(type, args, kwargs);
		};
	}
	if constexpr (HasInit<Type>) {
		type_prototype->__init__ =
			+[](PyObject *self, PyTuple *args, PyDict *kwargs) -> std::optional<int32_t> {
			return static_cast<Type *>(self)->__init__(args, kwargs);
		};
	}
	if constexpr (HasHash<Type>) {
		type_prototype->__hash__ = +[](const PyObject *self) -> size_t {
			return static_cast<const Type *>(self)->__hash__();
		};
	}
	if constexpr (HasLt<Type>) {
		type_prototype->__lt__ = +[](const PyObject *self, const PyObject *other) -> PyObject * {
			return static_cast<const Type *>(self)->__lt__(other);
		};
	}
	if constexpr (HasLe<Type>) {
		type_prototype->__le__ = +[](const PyObject *self, const PyObject *other) -> PyObject * {
			return static_cast<const Type *>(self)->__le__(other);
		};
	}
	if constexpr (HasEq<Type>) {
		type_prototype->__eq__ = +[](const PyObject *self, const PyObject *other) -> PyObject * {
			return static_cast<const Type *>(self)->__eq__(other);
		};
	}
	if constexpr (HasNe<Type>) {
		type_prototype->__ne__ = +[](const PyObject *self, const PyObject *other) -> PyObject * {
			return static_cast<const Type *>(self)->__ne__(other);
		};
	}
	if constexpr (HasGt<Type>) {
		type_prototype->__gt__ = +[](const PyObject *self, const PyObject *other) -> PyObject * {
			return static_cast<const Type *>(self)->__gt__(other);
		};
	}
	if constexpr (HasGe<Type>) {
		type_prototype->__ge__ = +[](const PyObject *self, const PyObject *other) -> PyObject * {
			return static_cast<const Type *>(self)->__ge__(other);
		};
	}
	if constexpr (HasIter<Type>) {
		type_prototype->__iter__ = +[](const PyObject *self) -> PyObject * {
			return static_cast<const Type *>(self)->__iter__();
		};
	}
	if constexpr (HasNext<Type>) {
		type_prototype->__next__ =
			+[](PyObject *self) -> PyObject * { return static_cast<Type *>(self)->__next__(); };
	}
	if constexpr (HasLength<Type>) {
		type_prototype->__len__ = +[](const PyObject *self) -> PyObject * {
			return static_cast<const Type *>(self)->__len__();
		};
	}
	if constexpr (HasAdd<Type>) {
		type_prototype->__add__ = +[](const PyObject *self, const PyObject *other) -> PyObject * {
			return static_cast<const Type *>(self)->__add__(other);
		};
	}
	if constexpr (HasSub<Type>) {
		type_prototype->__sub__ = +[](const PyObject *self, const PyObject *other) -> PyObject * {
			return static_cast<const Type *>(self)->__sub__(other);
		};
	}
	if constexpr (HasMul<Type>) {
		type_prototype->__mul__ = +[](const PyObject *self, const PyObject *other) -> PyObject * {
			return static_cast<const Type *>(self)->__mul__(other);
		};
	}
	if constexpr (HasExp<Type>) {
		type_prototype->__exp__ = +[](const PyObject *self, const PyObject *other) -> PyObject * {
			return static_cast<const Type *>(self)->__exp__(other);
		};
	}
	if constexpr (HasLshift<Type>) {
		type_prototype->__lshift__ =
			+[](const PyObject *self, const PyObject *other) -> PyObject * {
			return static_cast<const Type *>(self)->__lshift__(other);
		};
	}
	if constexpr (HasModulo<Type>) {
		type_prototype->__mod__ = +[](const PyObject *self, const PyObject *other) -> PyObject * {
			return static_cast<const Type *>(self)->__mod__(other);
		};
	}
	if constexpr (HasAbs<Type>) {
		type_prototype->__abs__ = +[](const PyObject *self) -> PyObject * {
			return static_cast<const Type *>(self)->__abs__();
		};
	}
	if constexpr (HasNeg<Type>) {
		type_prototype->__neg__ = +[](const PyObject *self) -> PyObject * {
			return static_cast<const Type *>(self)->__neg__();
		};
	}
	if constexpr (HasPos<Type>) {
		type_prototype->__pos__ = +[](const PyObject *self) -> PyObject * {
			return static_cast<const Type *>(self)->__pos__();
		};
	}
	if constexpr (HasInvert<Type>) {
		type_prototype->__invert__ = +[](const PyObject *self) -> PyObject * {
			return static_cast<const Type *>(self)->__invert__();
		};
	}
	if constexpr (HasBool<Type>) {
		type_prototype->__bool__ = +[](const PyObject *self) -> PyObject * {
			return static_cast<const Type *>(self)->__bool__();
		};
	}
	return type_prototype;
}


class PyBaseObject : public PyObject
{
  public:
	PyBaseObject(const TypePrototype &type) : PyObject(type) {}
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
