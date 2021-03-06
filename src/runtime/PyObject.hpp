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
	std::function<
		PyResult<PyObject *>(PyObject * /* self */, PyTuple * /* args */, PyDict * /* kwargs */)>
		method;
};

struct MemberDefinition
{
	std::string name;
	std::function<PyObject *(PyObject *)> member_accessor;
};

using CallSlotFunctionType = std::function<PyResult<PyObject *>(PyObject *, PyTuple *, PyDict *)>;
using NewSlotFunctionType =
	std::function<PyResult<PyObject *>(const PyType *, PyTuple *, PyDict *)>;
using InitSlotFunctionType = std::function<PyResult<int32_t>(PyObject *, PyTuple *, PyDict *)>;

using GetAttroFunctionType = std::function<PyResult<PyObject *>(const PyObject *, PyObject *)>;
using SetAttroFunctionType =
	std::function<PyResult<std::monostate>(PyObject *, PyObject *, PyObject *)>;

using GetSlotFunctionType =
	std::function<PyResult<PyObject *>(const PyObject *, PyObject *, PyObject *)>;
using SetSlotFunctionType =
	std::function<PyResult<std::monostate>(PyObject *, PyObject *, PyObject *)>;

using LenSlotFunctionType = std::function<PyResult<size_t>(const PyObject *)>;
using BoolSlotFunctionType = std::function<PyResult<bool>(const PyObject *)>;
using ReprSlotFunctionType = std::function<PyResult<PyObject *>(const PyObject *)>;
using IterSlotFunctionType = std::function<PyResult<PyObject *>(const PyObject *)>;
using NextSlotFunctionType = std::function<PyResult<PyObject *>(PyObject *)>;

using AbsSlotFunctionType = std::function<PyResult<PyObject *>(const PyObject *)>;
using NegSlotFunctionType = std::function<PyResult<PyObject *>(const PyObject *)>;
using PosSlotFunctionType = std::function<PyResult<PyObject *>(const PyObject *)>;
using InvertSlotFunctionType = std::function<PyResult<PyObject *>(const PyObject *)>;

using AddSlotFunctionType = std::function<PyResult<PyObject *>(const PyObject *, const PyObject *)>;
using SubtractSlotFunctionType =
	std::function<PyResult<PyObject *>(const PyObject *, const PyObject *)>;
using MultiplySlotFunctionType =
	std::function<PyResult<PyObject *>(const PyObject *, const PyObject *)>;
using ExpSlotFunctionType = std::function<PyResult<PyObject *>(const PyObject *, const PyObject *)>;
using LeftShiftSlotFunctionType =
	std::function<PyResult<PyObject *>(const PyObject *, const PyObject *)>;
using ModuloSlotFunctionType =
	std::function<PyResult<PyObject *>(const PyObject *, const PyObject *)>;

using HashSlotFunctionType = std::function<PyResult<size_t>(const PyObject *)>;
using CompareSlotFunctionType =
	std::function<PyResult<PyObject *>(const PyObject *, const PyObject *)>;

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

	std::vector<MemberDefinition> __members__;
	std::vector<MethodDefinition> __methods__;

	PyDict *__dict__{ nullptr };

	PyTuple *__mro__{ nullptr };
	PyTuple *__bases__{ nullptr };
	std::optional<TraverseFunctionType> traverse;

	template<typename Type> static std::unique_ptr<TypePrototype> create(std::string_view name);

	void add_member(MemberDefinition &&member) { __members__.push_back(std::move(member)); }
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

enum class LookupAttrResult { NOT_FOUND = 0, FOUND = 1 };

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
	PyObject() = delete;
	PyObject(const TypePrototype &type);

	virtual ~PyObject() = default;

	virtual PyType *type() const;

	template<typename T> static PyResult<PyObject *> from(const T &value);

	void visit_graph(Visitor &) override;

	PyResult<PyObject *> getattribute(PyObject *attribute) const;
	PyResult<std::monostate> setattribute(PyObject *attribute, PyObject *value);
	PyResult<PyObject *> get(PyObject *instance, PyObject *owner) const;

	PyResult<PyObject *> add(const PyObject *other) const;
	PyResult<PyObject *> subtract(const PyObject *other) const;
	PyResult<PyObject *> multiply(const PyObject *other) const;
	PyResult<PyObject *> exp(const PyObject *other) const;
	PyResult<PyObject *> lshift(const PyObject *other) const;
	PyResult<PyObject *> modulo(const PyObject *other) const;

	PyResult<PyObject *> neg() const;
	PyResult<PyObject *> pos() const;
	PyResult<PyObject *> abs() const;
	PyResult<PyObject *> invert() const;

	PyResult<PyObject *> repr() const;

	PyResult<size_t> hash() const;

	PyResult<PyObject *> richcompare(const PyObject *other, RichCompare) const;
	PyResult<PyObject *> eq(const PyObject *other) const;
	PyResult<PyObject *> ge(const PyObject *other) const;
	PyResult<PyObject *> gt(const PyObject *other) const;
	PyResult<PyObject *> le(const PyObject *other) const;
	PyResult<PyObject *> lt(const PyObject *other) const;
	PyResult<PyObject *> ne(const PyObject *other) const;

	PyResult<bool> bool_() const;
	PyResult<size_t> len() const;
	PyResult<PyObject *> iter() const;
	PyResult<PyObject *> next();

	PyResult<PyObject *> call(PyTuple *args, PyDict *kwargs);
	virtual PyResult<PyObject *> new_(PyTuple *args, PyDict *kwargs) const;
	PyResult<int32_t> init(PyTuple *args, PyDict *kwargs);

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __getattribute__(PyObject *attribute) const;
	PyResult<std::monostate> __setattribute__(PyObject *attribute, PyObject *value);
	PyResult<PyObject *> __eq__(const PyObject *other) const;
	PyResult<PyObject *> __repr__() const;
	PyResult<size_t> __hash__() const;
	PyResult<bool> __bool__() const;

	bool is_pyobject() const override { return true; }
	bool is_callable() const;
	const std::string &name() const;
	const TypePrototype &type_prototype() const { return m_type_prototype; }
	const PyDict &attributes() const { return *m_attributes; }
	PyResult<PyObject *> get_method(PyObject *name) const;
	PyResult<PyObject *> get_attribute(PyObject *name) const;
	std::tuple<PyResult<PyObject *>, LookupAttrResult> lookup_attribute(PyObject *name) const;

	static std::unique_ptr<TypePrototype> register_type();

	std::string to_string() const override;
};

template<typename Type> std::unique_ptr<TypePrototype> TypePrototype::create(std::string_view name)
{
	using namespace concepts;

	auto type_prototype = std::make_unique<TypePrototype>();
	type_prototype->__name__ = std::string(name);
	if constexpr (HasRepr<Type>) {
		type_prototype->__repr__ = +[](const PyObject *self) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__repr__();
		};
	}
	if constexpr (HasCall<Type>) {
		type_prototype->__call__ =
			+[](PyObject *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
			return static_cast<Type *>(self)->__call__(args, kwargs);
		};
	}
	if constexpr (HasNew<Type>) {
		type_prototype->__new__ =
			+[](const PyType *type, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
			return Type::__new__(type, args, kwargs);
		};
	}
	if constexpr (HasInit<Type>) {
		type_prototype->__init__ =
			+[](PyObject *self, PyTuple *args, PyDict *kwargs) -> PyResult<int32_t> {
			return static_cast<Type *>(self)->__init__(args, kwargs);
		};
	}
	if constexpr (HasHash<Type>) {
		type_prototype->__hash__ = +[](const PyObject *self) -> PyResult<size_t> {
			return static_cast<const Type *>(self)->__hash__();
		};
	}
	if constexpr (HasLt<Type>) {
		type_prototype->__lt__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__lt__(other);
		};
	}
	if constexpr (HasLe<Type>) {
		type_prototype->__le__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__le__(other);
		};
	}
	if constexpr (HasEq<Type>) {
		type_prototype->__eq__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__eq__(other);
		};
	}
	if constexpr (HasNe<Type>) {
		type_prototype->__ne__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__ne__(other);
		};
	}
	if constexpr (HasGt<Type>) {
		type_prototype->__gt__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__gt__(other);
		};
	}
	if constexpr (HasGe<Type>) {
		type_prototype->__ge__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__ge__(other);
		};
	}
	if constexpr (HasIter<Type>) {
		type_prototype->__iter__ = +[](const PyObject *self) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__iter__();
		};
	}
	if constexpr (HasNext<Type>) {
		type_prototype->__next__ = +[](PyObject *self) -> PyResult<PyObject *> {
			return static_cast<Type *>(self)->__next__();
		};
	}
	if constexpr (HasLength<Type>) {
		type_prototype->__len__ = +[](const PyObject *self) -> PyResult<size_t> {
			return static_cast<const Type *>(self)->__len__();
		};
	}
	if constexpr (HasAdd<Type>) {
		type_prototype->__add__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__add__(other);
		};
	}
	if constexpr (HasSub<Type>) {
		type_prototype->__sub__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__sub__(other);
		};
	}
	if constexpr (HasMul<Type>) {
		type_prototype->__mul__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__mul__(other);
		};
	}
	if constexpr (HasExp<Type>) {
		type_prototype->__exp__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__exp__(other);
		};
	}
	if constexpr (HasLshift<Type>) {
		type_prototype->__lshift__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__lshift__(other);
		};
	}
	if constexpr (HasModulo<Type>) {
		type_prototype->__mod__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__mod__(other);
		};
	}
	if constexpr (HasAbs<Type>) {
		type_prototype->__abs__ = +[](const PyObject *self) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__abs__();
		};
	}
	if constexpr (HasNeg<Type>) {
		type_prototype->__neg__ = +[](const PyObject *self) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__neg__();
		};
	}
	if constexpr (HasPos<Type>) {
		type_prototype->__pos__ = +[](const PyObject *self) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__pos__();
		};
	}
	if constexpr (HasInvert<Type>) {
		type_prototype->__invert__ = +[](const PyObject *self) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__invert__();
		};
	}
	if constexpr (HasBool<Type>) {
		type_prototype->__bool__ = +[](const PyObject *self) -> PyResult<bool> {
			return static_cast<const Type *>(self)->__bool__();
		};
	}
	if constexpr (HasGetAttro<Type>) {
		type_prototype->__getattribute__ =
			+[](const PyObject *self, PyObject *attr) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__getattribute__(attr);
		};
	}
	if constexpr (HasSetAttro<Type>) {
		type_prototype->__setattribute__ =
			+[](PyObject *self, PyObject *attr, PyObject *value) -> PyResult<std::monostate> {
			return static_cast<Type *>(self)->__setattribute__(attr, value);
		};
	}
	if constexpr (HasGet<Type>) {
		type_prototype->__get__ =
			+[](const PyObject *self, PyObject *instance, PyObject *owner) -> PyResult<PyObject *> {
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

}// namespace py