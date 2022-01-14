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

namespace py {

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

using GetAttroFunctionType = std::function<PyObject *(const PyObject *, PyObject *)>;
using SetAttroFunctionType = std::function<PyObject *(PyObject *, PyObject *, PyObject *)>;

using GetSlotFunctionType = std::function<PyObject *(const PyObject *, PyObject *, PyObject *)>;
using SetSlotFunctionType = std::function<bool(PyObject *, PyObject *, PyObject *)>;

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

using TraverseFunctionType = std::function<void(PyObject *, Cell::Visitor &)>;

struct TypePrototype
{
	std::string __name__;
	std::optional<std::variant<NewSlotFunctionType, PyObject *>> __new__;
	std::optional<std::variant<InitSlotFunctionType, PyObject *>> __init__;
	PyType *__class__{ nullptr };

	std::optional<std::variant<GetAttroFunctionType, PyObject *>> __getattribute__;
	std::optional<std::variant<SetAttroFunctionType, PyObject *>> __setattribute__;

	std::optional<std::variant<GetSlotFunctionType, PyObject *>> __get__;
	std::optional<std::variant<SetSlotFunctionType, PyObject *>> __set__;

	std::optional<std::variant<AddSlotFunctionType, PyObject *>> __add__;
	std::optional<std::variant<SubtractSlotFunctionType, PyObject *>> __sub__;
	std::optional<std::variant<MultiplySlotFunctionType, PyObject *>> __mul__;
	std::optional<std::variant<ExpSlotFunctionType, PyObject *>> __exp__;
	std::optional<std::variant<LeftShiftSlotFunctionType, PyObject *>> __lshift__;
	std::optional<std::variant<ModuloSlotFunctionType, PyObject *>> __mod__;

	std::optional<std::variant<AbsSlotFunctionType, PyObject *>> __abs__;
	std::optional<std::variant<NegSlotFunctionType, PyObject *>> __neg__;
	std::optional<std::variant<PosSlotFunctionType, PyObject *>> __pos__;
	std::optional<std::variant<InvertSlotFunctionType, PyObject *>> __invert__;

	std::optional<std::variant<CallSlotFunctionType, PyObject *>> __call__;
	std::optional<std::variant<LenSlotFunctionType, PyObject *>> __len__;
	std::optional<std::variant<BoolSlotFunctionType, PyObject *>> __bool__;
	std::optional<std::variant<ReprSlotFunctionType, PyObject *>> __repr__;
	std::optional<std::variant<IterSlotFunctionType, PyObject *>> __iter__;
	std::optional<std::variant<NextSlotFunctionType, PyObject *>> __next__;
	std::optional<std::variant<HashSlotFunctionType, PyObject *>> __hash__;

	std::optional<std::variant<CompareSlotFunctionType, PyObject *>> __eq__;
	std::optional<std::variant<CompareSlotFunctionType, PyObject *>> __gt__;
	std::optional<std::variant<CompareSlotFunctionType, PyObject *>> __ge__;
	std::optional<std::variant<CompareSlotFunctionType, PyObject *>> __le__;
	std::optional<std::variant<CompareSlotFunctionType, PyObject *>> __lt__;
	std::optional<std::variant<CompareSlotFunctionType, PyObject *>> __ne__;

	std::vector<MethodDefinition> __methods__;
	PyDict *__dict__{ nullptr };

	PyTuple *__mro__{ nullptr };
	PyTuple *__bases__{ nullptr };
	std::optional<TraverseFunctionType> traverse;

	template<typename Type> static std::unique_ptr<TypePrototype> create(std::string_view name);

	void add_method(MethodDefinition &&method) { __methods__.push_back(std::move(method)); }
};

namespace {
	template<typename T, typename... U>
	size_t get_address(const std::variant<std::function<T(U...)>, PyObject *> &f)
	{
		// adapted from https://stackoverflow.com/a/35920804
		if (std::holds_alternative<std::function<T(U...)>>(f)) {
			using FunctionType = T (*)(U...);
			auto fn_ptr = std::get<std::function<T(U...)>>(f).template target<FunctionType>();
			return bit_cast<size_t>(*fn_ptr);
		} else {
			// FIXME: is it valid to take this path? Is there use case?
			return bit_cast<size_t>(std::get<PyObject *>(f));
		}
	}
}// namespace

class PyObject : public Cell
{
	struct NotImplemented_
	{
	};

	friend class ::Heap;

  protected:
	const TypePrototype &m_type_prototype;
	PyDict *m_attributes{ nullptr };

  public:
	using PyResult = std::variant<PyObject *, NotImplemented_>;

	PyObject() = delete;
	PyObject(const TypePrototype &type);

	virtual ~PyObject() = default;

	virtual PyType *type() const;

	template<typename T> static PyObject *from(const T &value);

	void visit_graph(Visitor &) override;

	PyObject *getattribute(PyObject *attribute) const;
	bool setattribute(PyObject *attribute, PyObject *value);
	PyObject *get(PyObject *instance, PyObject *owner) const;

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

	static PyObject *__new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	std::optional<int32_t> __init__(PyTuple *args, PyDict *kwargs);

	PyObject *__getattribute__(PyObject *attribute) const;
	PyObject *__setattribute__(PyObject *attribute, PyObject *value);
	PyObject *__eq__(const PyObject *other) const;
	PyObject *__repr__() const;
	size_t __hash__() const;
	PyObject *__bool__() const;

	bool is_pyobject() const override { return true; }
	bool is_callable() const;
	const std::string &name() const;
	const TypePrototype &type_prototype() const { return m_type_prototype; }
	const PyDict &attributes() const { return *m_attributes; }
	PyObject *get_method(PyObject *name) const;
	PyObject *get_attribute(PyObject *name) const;

	static std::unique_ptr<TypePrototype> register_type();

	std::string to_string() const override;
};

template<typename Type> std::unique_ptr<TypePrototype> TypePrototype::create(std::string_view name)
{
	using namespace concepts;

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
	if constexpr (HasGetAttro<Type>) {
		type_prototype->__getattribute__ = +[](const PyObject *self, PyObject *attr) -> PyObject * {
			return static_cast<const Type *>(self)->__getattribute__(attr);
		};
	}
	if constexpr (HasSetAttro<Type>) {
		type_prototype->__setattribute__ =
			+[](PyObject *self, PyObject *attr, PyObject *value) -> PyObject * {
			return static_cast<Type *>(self)->__setattribute__(attr, value);
		};
	}
	if constexpr (HasGet<Type>) {
		type_prototype->__get__ =
			+[](const PyObject *self, PyObject *instance, PyObject *owner) -> PyObject * {
			return static_cast<const Type *>(self)->__get__(instance, owner);
		};
	}

	type_prototype->traverse =
		+[](PyObject *self, Cell::Visitor &visitor) { self->visit_graph(visitor); };

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

}// namespace py