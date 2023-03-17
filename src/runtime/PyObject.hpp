#pragma once

#include "../utilities.hpp"
#include "Value.hpp"
#include "concepts.hpp"
#include "forward.hpp"
#include "memory/GarbageCollector.hpp"
#include "runtime/forward.hpp"
#include "vm/VM.hpp"

#include <functional>
#include <memory>
#include <optional>
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

class MethodFlags
{
	std::bitset<10> m_flags;

  public:
	enum class Flag {
		VARARGS = 0,
		KEYWORDS = 1,
		NOARGS = 2,
		O = 3,
		CLASSMETHOD = 4,
		STATICMETHOD = 5,
		COEXIST = 6,
		FASTCALL = 7,
		STACKLESS = 8,
		METHOD = 9,
	};

  public:
	template<typename... Args> static MethodFlags create(Args... args)
	{
		MethodFlags f;
		(f.m_flags.set(static_cast<uint8_t>(args)), ...);
		return f;
	}

	bool is_set(Flag f) const { return m_flags.test(static_cast<size_t>(f)); }

	const std::bitset<10> &flags() const { return m_flags; }
};

struct MethodDefinition
{
	std::string name;
	std::function<PyResult<PyObject *>(PyObject * /* self or cls */,
		PyTuple * /* args */,
		PyDict * /* kwargs */)>
		method;
	MethodFlags flags;
	std::string doc;
};

struct MemberDefinition
{
	std::string name;
	std::function<PyObject *(PyObject *)> member_accessor;
	std::function<PyResult<std::monostate>(PyObject *, PyObject *)> member_setter;
};

struct PropertyDefinition
{
	std::string name;
	std::optional<std::function<PyResult<PyObject *>(PyObject *)>> member_getter;
	std::optional<std::function<PyResult<std::monostate>(PyObject *, PyObject *)>> member_setter;
};

using CallSlotFunctionType = std::function<PyResult<PyObject *>(PyObject *, PyTuple *, PyDict *)>;
using StrSlotFunctionType = std::function<PyResult<PyObject *>(PyObject *)>;
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

using GetItemSlotFunctionType = std::function<PyResult<PyObject *>(PyObject *, PyObject *)>;
using SetItemSlotFunctionType =
	std::function<PyResult<std::monostate>(PyObject *, PyObject *, PyObject *)>;
using DelItemSlotFunctionType = std::function<PyResult<std::monostate>(PyObject *, PyObject *)>;

using GetItemSequenceSlotFunctionType = std::function<PyResult<PyObject *>(PyObject *, int64_t)>;
using SetItemSequenceSlotFunctionType =
	std::function<PyResult<std::monostate>(PyObject *, int64_t, PyObject *)>;
using DelItemSequenceSlotFunctionType =
	std::function<PyResult<std::monostate>(PyObject *, int64_t)>;

using LenSlotFunctionType = std::function<PyResult<size_t>(const PyObject *)>;
using BoolSlotFunctionType = std::function<PyResult<bool>(const PyObject *)>;
using ContainsSlotFunctionType = std::function<PyResult<bool>(PyObject *, PyObject *)>;
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
using FloorDivSlotFunctionType = std::function<PyResult<PyObject *>(PyObject *, PyObject *)>;
using TrueDivSlotFunctionType = std::function<PyResult<PyObject *>(PyObject *, PyObject *)>;
using LeftShiftSlotFunctionType =
	std::function<PyResult<PyObject *>(const PyObject *, const PyObject *)>;
using RightShiftSlotFunctionType =
	std::function<PyResult<PyObject *>(const PyObject *, const PyObject *)>;
using ModuloSlotFunctionType =
	std::function<PyResult<PyObject *>(const PyObject *, const PyObject *)>;
using AndSlotFunctionType = std::function<PyResult<PyObject *>(PyObject *, PyObject *)>;
using OrSlotFunctionType = std::function<PyResult<PyObject *>(PyObject *, PyObject *)>;

using HashSlotFunctionType = std::function<PyResult<int64_t>(const PyObject *)>;
using CompareSlotFunctionType =
	std::function<PyResult<PyObject *>(const PyObject *, const PyObject *)>;

using TraverseFunctionType = std::function<void(PyObject *, Cell::Visitor &)>;

struct NumberTypePrototype
{
};

struct MappingTypePrototype
{
	std::optional<std::variant<LenSlotFunctionType, PyObject *>> __len__;
	std::optional<std::variant<GetItemSlotFunctionType, PyObject *>> __getitem__;
	std::optional<std::variant<SetItemSlotFunctionType, PyObject *>> __setitem__;
	std::optional<std::variant<DelItemSlotFunctionType, PyObject *>> __delitem__;
};

struct SequenceTypePrototype
{
	std::optional<std::variant<LenSlotFunctionType, PyObject *>> __len__;
	std::optional<std::variant<AddSlotFunctionType, PyObject *>> __concat__;
	std::optional<std::variant<GetItemSequenceSlotFunctionType, PyObject *>> __getitem__;
	std::optional<std::variant<SetItemSequenceSlotFunctionType, PyObject *>> __setitem__;
	std::optional<std::variant<DelItemSequenceSlotFunctionType, PyObject *>> __delitem__;
	std::optional<std::variant<ContainsSlotFunctionType, PyObject *>> __contains__;
};

struct PyBuffer
{
	Bytes buf;
	PyObject *obj;
	int64_t len;
	int64_t itemsize;
	int readonly;
	int ndim;
	std::string format;
	std::vector<int64_t> shape;
	std::vector<int64_t> strides;
	std::vector<int64_t> suboffsets;
	void *internal;

	bool is_ccontiguous() const { return len == 0 || strides.empty(); }

	bool is_contiguous(char order) const
	{
		if (!suboffsets.empty()) return false;
		if (order == 'C') {
			return is_ccontiguous();
		} else {
			TODO();
		}
		return false;
	}
};

struct PyBufferProcs
{
	std::function<PyResult<std::monostate>(PyObject *, PyBuffer &, int)> getbuffer;
	std::function<PyResult<std::monostate>(PyObject *, PyBuffer &)> releasebuffer;
};

struct TypePrototype
{
  private:
	TypePrototype(const TypePrototype &) = default;
	TypePrototype &operator=(const TypePrototype &) = default;

  public:
	TypePrototype() = default;

	std::unique_ptr<TypePrototype> clone() const
	{
		return std::unique_ptr<TypePrototype>(new TypePrototype{ *this });
	}

  public:
	std::string __name__;
	PyType *__base__{ nullptr };
	PyTuple *__bases__{ nullptr };

	std::function<PyResult<PyObject *>(PyType *)> __alloc__;
	std::optional<std::variant<NewSlotFunctionType, PyObject *>> __new__;
	std::optional<std::variant<InitSlotFunctionType, PyObject *>> __init__;
	PyType *__class__{ nullptr };
	std::optional<std::string> __doc__;

	std::optional<NumberTypePrototype> number_type_protocol;
	std::optional<MappingTypePrototype> mapping_type_protocol;
	std::optional<SequenceTypePrototype> sequence_type_protocol;

	std::optional<std::variant<GetAttroFunctionType, PyObject *>> __getattribute__;
	std::optional<std::variant<SetAttroFunctionType, PyObject *>> __setattribute__;

	std::optional<std::variant<GetSlotFunctionType, PyObject *>> __get__;
	std::optional<std::variant<SetSlotFunctionType, PyObject *>> __set__;

	std::optional<std::variant<AddSlotFunctionType, PyObject *>> __add__;
	std::optional<std::variant<SubtractSlotFunctionType, PyObject *>> __sub__;
	std::optional<std::variant<MultiplySlotFunctionType, PyObject *>> __mul__;
	std::optional<std::variant<ExpSlotFunctionType, PyObject *>> __exp__;
	std::optional<std::variant<TrueDivSlotFunctionType, PyObject *>> __truediv__;
	std::optional<std::variant<FloorDivSlotFunctionType, PyObject *>> __floordiv__;
	std::optional<std::variant<LeftShiftSlotFunctionType, PyObject *>> __lshift__;
	std::optional<std::variant<RightShiftSlotFunctionType, PyObject *>> __rshift__;
	std::optional<std::variant<ModuloSlotFunctionType, PyObject *>> __mod__;
	std::optional<std::variant<AndSlotFunctionType, PyObject *>> __and__;
	std::optional<std::variant<OrSlotFunctionType, PyObject *>> __or__;

	std::optional<std::variant<AbsSlotFunctionType, PyObject *>> __abs__;
	std::optional<std::variant<NegSlotFunctionType, PyObject *>> __neg__;
	std::optional<std::variant<PosSlotFunctionType, PyObject *>> __pos__;
	std::optional<std::variant<InvertSlotFunctionType, PyObject *>> __invert__;

	std::optional<std::variant<CallSlotFunctionType, PyObject *>> __call__;
	std::optional<std::variant<StrSlotFunctionType, PyObject *>> __str__;
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

	std::optional<PyBufferProcs> as_buffer;

	std::vector<MemberDefinition> __members__;
	std::vector<PropertyDefinition> __getset__;
	std::vector<MethodDefinition> __methods__;

	PyDict *__dict__{ nullptr };

	std::optional<TraverseFunctionType> traverse;

	bool is_ready{ false };
	bool is_heaptype{ false };
	bool is_type{ false };

	template<typename Type> static std::unique_ptr<TypePrototype> create(std::string_view name);

	void add_member(MemberDefinition &&member) { __members__.push_back(std::move(member)); }
	void add_property(PropertyDefinition &&property) { __getset__.push_back(std::move(property)); }
	void add_method(MethodDefinition &&method) { __methods__.push_back(std::move(method)); }

	void visit_graph(::Cell::Visitor &visitor);
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

class PyMappingWrapper
{
	PyObject *m_object;

  public:
	PyMappingWrapper(PyObject *object) : m_object(object) {}
	PyResult<size_t> len();
	PyResult<PyObject *> getitem(PyObject *);// mp_subscript
	PyResult<std::monostate> setitem(PyObject *, PyObject *);// mp_ass_subscript
	PyResult<std::monostate> delitem(PyObject *);// mp_ass_subscript
};

struct PySequence
{
};

class PySequenceWrapper
{
	PyObject *m_object;

  public:
	PySequenceWrapper(PyObject *object) : m_object(object) {}
	PyResult<size_t> len();
	PyResult<PyObject *> concat(PyObject *other);
	PyResult<PyObject *> getitem(int64_t);// sq_item
	PyResult<std::monostate> setitem(int64_t, PyObject *);// sq_ass_item
	PyResult<std::monostate> delitem(int64_t);// sq_ass_item
	PyResult<bool> contains(PyObject *);
};

class PyObject : public Cell
{
	friend class ::Heap;
	friend PyMappingWrapper;
	friend PySequenceWrapper;
	friend class PyType;

  protected:
	std::variant<std::reference_wrapper<const TypePrototype>, PyType *> m_type;
	PyDict *m_attributes{ nullptr };

  public:
	PyObject() = delete;
	PyObject(const TypePrototype &type);
	PyObject(PyType *type);

	virtual ~PyObject() = default;

	virtual PyType *static_type() const;
	PyType *type() const;

	template<typename T> static PyResult<PyObject *> from(const T &value);

	void visit_graph(Visitor &) override;

	PyResult<PyMappingWrapper> as_mapping();
	PyResult<PySequenceWrapper> as_sequence();
	PyResult<PyBufferProcs> as_buffer();

	PyResult<std::monostate> get_buffer(PyBuffer &, int flags);

	PyResult<PyObject *> getattribute(PyObject *attribute) const;
	PyResult<std::monostate> setattribute(PyObject *attribute, PyObject *value);
	PyResult<PyObject *> get(PyObject *instance, PyObject *owner) const;

	PyResult<PyObject *> add(const PyObject *other) const;
	PyResult<PyObject *> subtract(const PyObject *other) const;
	PyResult<PyObject *> multiply(const PyObject *other) const;
	PyResult<PyObject *> exp(const PyObject *other) const;
	PyResult<PyObject *> floordiv(PyObject *);
	PyResult<PyObject *> truediv(PyObject *);
	PyResult<PyObject *> lshift(const PyObject *other) const;
	PyResult<PyObject *> rshift(const PyObject *other) const;
	PyResult<PyObject *> modulo(const PyObject *other) const;
	PyResult<PyObject *> and_(PyObject *other);
	PyResult<PyObject *> or_(PyObject *other);

	PyResult<bool> contains(PyObject *value);
	PyResult<std::monostate> delete_item(PyObject *key);

	PyResult<PyObject *> neg() const;
	PyResult<PyObject *> pos() const;
	PyResult<PyObject *> abs() const;
	PyResult<PyObject *> invert() const;

	PyResult<PyString *> repr() const;
	PyResult<PyString *> str();

	PyResult<int64_t> hash() const;

	PyResult<PyObject *> richcompare(const PyObject *other, RichCompare) const;
	PyResult<PyObject *> eq(const PyObject *other) const;
	PyResult<PyObject *> ge(const PyObject *other) const;
	PyResult<PyObject *> gt(const PyObject *other) const;
	PyResult<PyObject *> le(const PyObject *other) const;
	PyResult<PyObject *> lt(const PyObject *other) const;
	PyResult<PyObject *> ne(const PyObject *other) const;

	virtual PyResult<bool> true_();
	PyResult<PyObject *> iter() const;
	PyResult<PyObject *> next();

	PyResult<PyObject *> call(PyTuple *args, PyDict *kwargs);
	virtual PyResult<PyObject *> new_(PyTuple *args, PyDict *kwargs) const;
	PyResult<int32_t> init(PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> getitem(PyObject *key);
	PyResult<std::monostate> setitem(PyObject *key, PyObject *value);
	PyResult<std::monostate> delitem(PyObject *key);

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<int32_t> __init__(PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> __getattribute__(PyObject *attribute) const;
	PyResult<std::monostate> __setattribute__(PyObject *attribute, PyObject *value);
	PyResult<PyObject *> __eq__(const PyObject *other) const;
	PyResult<PyObject *> __ne__(const PyObject *other) const;
	PyResult<PyObject *> __repr__() const;
	PyResult<int64_t> __hash__() const;
	PyResult<PyObject *> __str__();

	bool is_pyobject() const override { return true; }
	bool is_callable() const;
	const std::string &name() const;
	const TypePrototype &type_prototype() const;
	const PyDict *attributes() const { return m_attributes; }
	PyDict *attributes() { return m_attributes; }
	PyResult<PyObject *> get_method(PyObject *name) const;
	PyResult<PyObject *> get_attribute(PyObject *name) const;
	std::tuple<PyResult<PyObject *>, LookupAttrResult> lookup_attribute(PyObject *name) const;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	std::string to_string() const override;
};

// avoid explicit specialization after instantiations
template<> PyResult<PyObject *> PyObject::from(PyObject *const &value);
template<> PyResult<PyObject *> PyObject::from(const Number &value);
template<> PyResult<PyObject *> PyObject::from(const int64_t &value);
template<> PyResult<PyObject *> PyObject::from(const String &value);
template<> PyResult<PyObject *> PyObject::from(const Bytes &value);
template<> PyResult<PyObject *> PyObject::from(const Ellipsis &value);
template<> PyResult<PyObject *> PyObject::from(const NameConstant &value);
template<> PyResult<PyObject *> PyObject::from(const Value &value);

BaseException *memory_error(size_t failed_allocation_size);

template<typename Type> std::unique_ptr<TypePrototype> TypePrototype::create(std::string_view name)
{
	using namespace concepts;

	auto type_prototype = std::make_unique<TypePrototype>();
	type_prototype->__name__ = std::string(name);
	type_prototype->__alloc__ = [](PyType *t) -> PyResult<PyObject *> {
		auto *obj = VirtualMachine::the().heap().allocate<Type>(t);
		if (!obj) { return Err(memory_error(sizeof(Type))); }
		return Ok(obj);
	};
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
	if constexpr (HasDoc<Type>) { type_prototype->__doc__ = Type::__doc__; }
	if constexpr (HasHash<Type>) {
		type_prototype->__hash__ = +[](const PyObject *self) -> PyResult<int64_t> {
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
		if (!type_prototype->mapping_type_protocol.has_value()) {
			type_prototype->mapping_type_protocol =
				MappingTypePrototype{ .__len__ = +[](const PyObject *self) -> PyResult<size_t> {
					return static_cast<const Type *>(self)->__len__();
				} };
		} else {
			type_prototype->mapping_type_protocol->__len__ =
				+[](const PyObject *self) -> PyResult<size_t> {
				return static_cast<const Type *>(self)->__len__();
			};
		}
	}
	if constexpr (HasSetItem<Type>) {
		if (!type_prototype->mapping_type_protocol.has_value()) {
			type_prototype->mapping_type_protocol =
				MappingTypePrototype{ .__setitem__ =
										  +[](PyObject *self,
											   PyObject *name,
											   PyObject *value) -> PyResult<std::monostate> {
					return static_cast<Type *>(self)->__setitem__(name, value);
				} };
		} else {
			type_prototype->mapping_type_protocol->__setitem__ =
				+[](PyObject *self, PyObject *name, PyObject *value) -> PyResult<std::monostate> {
				return static_cast<Type *>(self)->__setitem__(name, value);
			};
		}
	}
	if constexpr (HasGetItem<Type>) {
		if (!type_prototype->mapping_type_protocol.has_value()) {
			type_prototype->mapping_type_protocol = MappingTypePrototype{
				.__getitem__ = +[](PyObject *self, PyObject *name) -> PyResult<PyObject *> {
					return static_cast<Type *>(self)->__getitem__(name);
				}
			};
		} else {
			type_prototype->mapping_type_protocol->__getitem__ =
				+[](PyObject *self, PyObject *name) -> PyResult<PyObject *> {
				return static_cast<Type *>(self)->__getitem__(name);
			};
		}
	}
	if constexpr (HasDelItem<Type>) {
		if (!type_prototype->mapping_type_protocol.has_value()) {
			type_prototype->mapping_type_protocol = MappingTypePrototype{
				.__delitem__ = +[](PyObject *self, PyObject *name) -> PyResult<std::monostate> {
					return static_cast<Type *>(self)->__delitem__(name);
				}
			};
		} else {
			type_prototype->mapping_type_protocol->__delitem__ =
				+[](PyObject *self, PyObject *name) -> PyResult<std::monostate> {
				return static_cast<Type *>(self)->__delitem__(name);
			};
		}
	}
	if constexpr (HasContains<Type>) {
		if (!type_prototype->sequence_type_protocol.has_value()) {
			type_prototype->sequence_type_protocol =
				SequenceTypePrototype{ .__contains__ =
										   +[](PyObject *self, PyObject *value) -> PyResult<bool> {
					return static_cast<Type *>(self)->__contains__(value);
				} };
		} else {
			type_prototype->sequence_type_protocol->__contains__ =
				+[](PyObject *self, PyObject *value) -> PyResult<bool> {
				return static_cast<Type *>(self)->__contains__(value);
			};
		}
	}
	if constexpr (HasSequenceGetItem<Type>) {
		if (!type_prototype->sequence_type_protocol.has_value()) {
			type_prototype->sequence_type_protocol = SequenceTypePrototype{
				.__getitem__ = +[](PyObject *self, int64_t index) -> PyResult<PyObject *> {
					return static_cast<Type *>(self)->__getitem__(index);
				}
			};
		} else {
			type_prototype->sequence_type_protocol->__getitem__ =
				+[](PyObject *self, int64_t index) -> PyResult<PyObject *> {
				return static_cast<Type *>(self)->__getitem__(index);
			};
		}
	}
	if constexpr (HasSequenceSetItem<Type>) {
		if (!type_prototype->sequence_type_protocol.has_value()) {
			type_prototype->sequence_type_protocol =
				SequenceTypePrototype{ .__setitem__ =
										   +[](PyObject *self,
												int64_t index,
												PyObject *value) -> PyResult<std::monostate> {
					return static_cast<Type *>(self)->__setitem__(index, value);
				} };
		} else {
			type_prototype->sequence_type_protocol->__setitem__ =
				+[](PyObject *self, int64_t index, PyObject *value) -> PyResult<std::monostate> {
				return static_cast<Type *>(self)->__setitem__(index, value);
			};
		}
	}
	if constexpr (HasSequenceDelItem<Type>) {
		if (!type_prototype->sequence_type_protocol.has_value()) {
			type_prototype->sequence_type_protocol = SequenceTypePrototype{
				.__delitem__ = +[](PyObject *self, int64_t index) -> PyResult<std::monostate> {
					return static_cast<Type *>(self)->__delitem__(index);
				}
			};
		} else {
			type_prototype->sequence_type_protocol->__delitem__ =
				+[](PyObject *self, int64_t index) -> PyResult<std::monostate> {
				return static_cast<Type *>(self)->__delitem__(index);
			};
		}
	}
	if constexpr (std::is_base_of_v<PySequence, Type>) {
		if (!type_prototype->sequence_type_protocol.has_value()) {
			type_prototype->sequence_type_protocol = SequenceTypePrototype{
				.__concat__ =
					+[](const PyObject *self, const PyObject *value) -> PyResult<PyObject *> {
					return static_cast<const Type *>(self)->__add__(value);
				}
			};
		} else {
			type_prototype->sequence_type_protocol->__concat__ =
				+[](const PyObject *self, const PyObject *value) -> PyResult<PyObject *> {
				return static_cast<const Type *>(self)->__add__(value);
			};
		}
	} else {
		if constexpr (HasAdd<Type>) {
			type_prototype->__add__ =
				+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
				return static_cast<const Type *>(self)->__add__(other);
			};
		}
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
	if constexpr (HasTrueDiv<Type>) {
		type_prototype->__truediv__ = +[](PyObject *self, PyObject *other) -> PyResult<PyObject *> {
			return static_cast<Type *>(self)->__truediv__(other);
		};
	}
	if constexpr (HasFloorDiv<Type>) {
		type_prototype->__floordiv__ =
			+[](PyObject *self, PyObject *other) -> PyResult<PyObject *> {
			return static_cast<Type *>(self)->__floordiv__(other);
		};
	}
	if constexpr (HasLshift<Type>) {
		type_prototype->__lshift__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__lshift__(other);
		};
	}
	if constexpr (HasRshift<Type>) {
		type_prototype->__rshift__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__rshift__(other);
		};
	}
	if constexpr (HasModulo<Type>) {
		type_prototype->__mod__ =
			+[](const PyObject *self, const PyObject *other) -> PyResult<PyObject *> {
			return static_cast<const Type *>(self)->__mod__(other);
		};
	}
	if constexpr (HasAnd<Type>) {
		type_prototype->__and__ = +[](PyObject *self, PyObject *other) -> PyResult<PyObject *> {
			return static_cast<Type *>(self)->__and__(other);
		};
	}
	if constexpr (HasOr<Type>) {
		type_prototype->__or__ = +[](PyObject *self, PyObject *other) -> PyResult<PyObject *> {
			return static_cast<Type *>(self)->__or__(other);
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
	if constexpr (HasSet<Type>) {
		type_prototype->__set__ =
			+[](PyObject *self, PyObject *attribute, PyObject *value) -> PyResult<std::monostate> {
			return static_cast<Type *>(self)->__set__(attribute, value);
		};
	}
	if constexpr (HasStr<Type>) {
		type_prototype->__str__ = +[](PyObject *self) -> PyResult<PyObject *> {
			return static_cast<Type *>(self)->__str__();
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
	PyBaseObject(PyType *type) : PyObject(type) {}
};

struct ValueHash
{
	size_t operator()(const Value &value) const;
};

}// namespace py
