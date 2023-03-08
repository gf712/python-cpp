#pragma once

#include "PyBool.hpp"
#include "PyInteger.hpp"
#include "PyNone.hpp"
#include "PyString.hpp"
#include "PyType.hpp"
#include "TypeError.hpp"

namespace py {

class Slot
{
  public:
	enum Flags {
		None = 0,
		Keywords = 1,
		New = 2,
	};

	std::string_view name;
	std::optional<std::string_view> doc;
	std::function<bool(TypePrototype &)> has_member;
	std::function<void(TypePrototype &)> reset_member;
	std::function<std::optional<std::variant<void *, PyObject *>>(TypePrototype &)> get_member;
	std::function<void(TypePrototype &, std::variant<void *, PyObject *>)> set_member;
	std::function<void(TypePrototype &, PyObject *)> update_member;
	std::function<PyResult<PySlotWrapper *>(PyType *)> create_slot_wrapper;
	const Flags flags;

  private:
	Slot() : flags(Flags::None) {}

	template<typename T> struct arity;

	template<typename R, typename... Args> struct arity<std::function<R(Args...)>>
	{
		static constexpr size_t value = sizeof...(Args);
		template<size_t Idx>
		using arg_type = typename std::tuple_element_t<Idx, std::tuple<Args...>>;
	};

	template<Slot::Flags flags_, typename TypePrototypeWrapper, typename R, typename... Args>
	constexpr void initialize(std::optional<std::variant<std::function<R(Args...)>, PyObject *>>
			TypePrototypeWrapper::*slot);

  public:
	template<typename FnType, typename TypePrototypeWrapper>
	constexpr Slot(std::string_view name_,
		std::optional<std::variant<std::function<FnType>, PyObject *>> TypePrototypeWrapper::*slot)
		: Slot(name_, slot, std::nullopt)
	{}

	template<typename FnType, typename TypePrototypeWrapper>
	constexpr Slot(std::string_view name_,
		std::optional<std::variant<std::function<FnType>, PyObject *>> TypePrototypeWrapper::*slot,
		std::optional<std::string_view> doc_)
		: name(name_), doc(doc_), flags(Flags::None)
	{
		initialize<Flags::None>(slot);
	}

	template<typename FnType>
	static Slot with_keyword(std::string_view name_,
		std::optional<std::variant<std::function<FnType>, PyObject *>> TypePrototype::*slot,
		std::optional<std::string_view> doc_)
	{
		Slot s{};
		s.name = name_;
		s.doc = std::move(doc_);
		s.initialize<Flags::Keywords>(slot);
		return s;
	}

	template<typename FnType>
	static Slot with_new(std::string_view name_,
		std::optional<std::variant<std::function<FnType>, PyObject *>> TypePrototype::*slot,
		std::optional<std::string_view> doc_)
	{
		Slot s{};
		s.name = name_;
		s.doc = std::move(doc_);
		s.initialize<Flags::New>(slot);
		return s;
	}
};

class PySlotWrapper : public PyBaseObject
{
	using FunctionType = std::function<PyResult<PyObject *>(PyObject *, PyTuple *, PyDict *)>;

	PyString *m_name{ nullptr };
	PyType *m_type{ nullptr };
	std::optional<std::reference_wrapper<Slot>> m_base;
	FunctionType m_slot;

	friend class ::Heap;

	PySlotWrapper(PyType *type);

	PySlotWrapper(PyString *name, PyType *underlying_type, Slot &base, FunctionType &&function);

  public:
	static PyResult<PySlotWrapper *>
		create(PyString *name, PyType *underlying_type, Slot &base, FunctionType &&function);

	PyString *slot_name() { return m_name; }
	FunctionType slot() { return m_slot; }
	const FunctionType &slot() const { return m_slot; }
	std::reference_wrapper<Slot> base() const
	{
		ASSERT(m_base);
		return m_base->get();
	}
	PyType *base_type() const { return m_type; }

	std::string to_string() const override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __get__(PyObject *, PyObject *) const;

	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

template<Slot::Flags flags_, typename TypePrototypeWrapper, typename R, typename... Args>
constexpr void Slot::initialize(
	std::optional<std::variant<std::function<R(Args...)>, PyObject *>> TypePrototypeWrapper::*slot)
{
	using FnType = std::function<R(Args...)>;
	using FnPointerType = typename std::add_pointer_t<R(Args...)>;

	auto get_slot = [](TypePrototype &t) -> TypePrototypeWrapper & {
		if constexpr (std::is_same_v<TypePrototypeWrapper, TypePrototype>) {
			return t;
		} else if constexpr (std::is_same_v<TypePrototypeWrapper, MappingTypePrototype>) {
			if (!t.mapping_type_protocol.has_value()) {
				t.mapping_type_protocol = MappingTypePrototype{};
			}
			return *t.mapping_type_protocol;
		} else if constexpr (std::is_same_v<TypePrototypeWrapper, SequenceTypePrototype>) {
			if (!t.sequence_type_protocol.has_value()) {
				t.sequence_type_protocol = SequenceTypePrototype{};
			}
			return *t.sequence_type_protocol;
		}
	};

	has_member = [slot, get_slot](
					 TypePrototype &t) -> bool { return (get_slot(t).*slot).has_value(); };
	reset_member = [slot, get_slot](TypePrototype &t) { get_slot(t).*slot = std::nullopt; };
	get_member = [slot, get_slot](
					 TypePrototype &t) -> std::optional<std::variant<void *, PyObject *>> {
		auto &slot_value = get_slot(t).*slot;
		if (slot_value.has_value()) {
			if (std::holds_alternative<FnType>(*slot_value)) {
				ASSERT(std::get<FnType>(*slot_value));
				auto fn_ptr = std::get<FnType>(*slot_value).template target<FnPointerType>();
				ASSERT(fn_ptr);
				return reinterpret_cast<void *>(*fn_ptr);
			} else {
				ASSERT(std::get<PyObject *>(*slot_value));
				return std::get<PyObject *>(*slot_value);
			}
		} else {
			return std::nullopt;
		}
	};
	set_member = [slot, get_slot](TypePrototype &t, std::variant<void *, PyObject *> fn) {
		auto &slot_value = get_slot(t).*slot;
		if (std::holds_alternative<void *>(fn)) {
			ASSERT(std::get<void *>(fn));
			slot_value = reinterpret_cast<FnPointerType>(std::get<void *>(fn));
		} else {
			ASSERT(std::get<PyObject *>(fn));
			slot_value = std::get<PyObject *>(fn);
		}
	};
	update_member = [slot, get_slot](TypePrototype &t, PyObject *obj) { get_slot(t).*slot = obj; };
	create_slot_wrapper = [this, slot, get_slot, name_ = name](
							  PyType *type) -> PyResult<PySlotWrapper *> {
		auto &t = type->underlying_type();
		auto &slot_value = get_slot(t).*slot;
		ASSERT(slot_value.has_value());
		auto &fn_ = *slot_value;
		auto name = PyString::create(std::string{ name_ });
		if (name.is_err()) { return Err(name.unwrap_err()); };
		if (std::holds_alternative<PyObject *>(fn_)) {
			auto fn = [fn_](PyObject *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
				std::vector<Value> new_args_vector;
				new_args_vector.reserve(args->size() + 1);
				new_args_vector.push_back(self);
				new_args_vector.insert(
					new_args_vector.end(), args->elements().begin(), args->elements().end());
				auto args_ = PyTuple::create(new_args_vector);
				if (args_.is_err()) return args_;
				args = args_.unwrap();
				if constexpr (flags_ == Flags::Keywords) {
					return std::get<PyObject *>(fn_)->call(args, kwargs);
				} else {
					return std::get<PyObject *>(fn_)->call(args, nullptr);
				}
			};
			return PySlotWrapper::create(name.unwrap(), type, *this, std::move(fn));
		} else {
			auto fn = [&fn_](
						  PyObject *self, PyTuple *args, PyDict *kwargs) -> PyResult<PyObject *> {
				auto &native_fn = std::get<FnType>(fn_);
				auto wrap_result = [](auto result) -> PyResult<PyObject *> {
					using ResultType = typename FnType::result_type;
					static_assert(detail::is_pyresult<ResultType>{},
						"Native functions should always return PyResult<T>");
					if constexpr (std::is_same_v<typename ResultType::OkType, PyObject *>) {
						return result;
					} else if constexpr (std::is_same_v<typename ResultType::OkType,
											 std::monostate>) {
						return result.and_then([](std::monostate) { return Ok(py_none()); });
					} else if constexpr (std::is_same_v<typename ResultType::OkType, bool>) {
						return result.and_then(
							[](bool value) { return Ok(value ? py_true() : py_false()); });
					} else if constexpr (std::is_integral_v<typename ResultType::OkType>) {
						return result.and_then([](auto value) { return PyInteger::create(value); });
					} else {
						[]<bool flag = false>()
						{
							static_assert(flag, "unsupported native return type");
						}
						();
					}
				};
				auto get_parameter = [&args]<typename T, size_t Idx>() -> PyResult<T> {
					using ArgType = typename std::remove_cv_t<std::remove_pointer_t<T>>;
					if constexpr (std::is_same_v<ArgType, PyObject>) {
						return PyObject::from(args->elements()[Idx]);
					} else if constexpr (std::is_same_v<ArgType, int64_t>) {
						const auto &el = args->elements()[Idx];
						if (std::holds_alternative<PyObject *>(el)) {
							auto obj = std::get<PyObject *>(el);
							if (auto int_obj = as<PyInteger>(obj)) { return Ok(int_obj->as_i64()); }
							return Err(type_error(
								"expected integer type, but got {}", obj->type()->name()));
						} else if (std::holds_alternative<Number>(el)) {
							if (std::holds_alternative<double>(std::get<Number>(el).value)) {
								return Err(type_error("expected integer type, but got float"));
							}
							return Ok(std::get<BigIntType>(std::get<Number>(el).value).get_si());
						} else {
							TODO();
						}
					} else {
						[]<bool flag = false>()
						{
							static_assert(flag, "unsupported native parameter type");
						}
						();
					}
				};
				if constexpr (flags_ == Flags::Keywords) {
					return wrap_result(native_fn(self, args, kwargs));
				} else if constexpr (flags_ == Flags::New) {
					auto type_ = as<PyType>(self);
					// this check should have been done by the caller
					ASSERT(type_);
					return wrap_result(native_fn(type_, args, kwargs));
				} else {
					if constexpr (arity<FnType>::value == 1) {
						return wrap_result(native_fn(self));
					} else if constexpr (arity<FnType>::value == 2) {
						ASSERT(args->elements().size() >= 1);
						using arg_type0 = typename arity<FnType>::template arg_type<1>;
						auto arg0 = get_parameter.template operator()<arg_type0, 0>();
						if (arg0.is_err()) { return Err(arg0.unwrap_err()); }
						return wrap_result(native_fn(self, arg0.unwrap()));
					} else if constexpr (arity<FnType>::value == 3) {
						ASSERT(args->elements().size() >= 2);
						using arg_type0 = typename arity<FnType>::template arg_type<1>;
						using arg_type1 = typename arity<FnType>::template arg_type<2>;
						auto arg0 = get_parameter.template operator()<arg_type0, 0>();
						auto arg1 = get_parameter.template operator()<arg_type1, 1>();
						if (arg0.is_err()) { return Err(arg0.unwrap_err()); }
						if (arg1.is_err()) { return Err(arg1.unwrap_err()); }
						return wrap_result(native_fn(self, arg0.unwrap(), arg1.unwrap()));
					} else {
						[]<bool flag = false>() { static_assert(flag, "unsupported arity"); }
						();
					}
				}
			};
			return PySlotWrapper::create(name.unwrap(), type, *this, std::move(fn));
		}
	};
}


}// namespace py
