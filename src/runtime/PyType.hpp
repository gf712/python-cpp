#pragma once

#include "PyObject.hpp"

namespace py {

class PyType : public PyBaseObject
{
	template<typename T> friend struct klass;
	friend class ::Heap;
	friend class PyObject;

  private:
	std::variant<std::reference_wrapper<TypePrototype>, std::unique_ptr<TypePrototype>>
		m_underlying_type;

  public:
	PyString *__name__{ nullptr };
	PyString *__qualname__{ nullptr };
	std::vector<PyObject *> __slots__;
	PyString *__module__{ nullptr };
	mutable PyTuple *__mro__{ nullptr };
	std::variant<std::monostate, std::reference_wrapper<TypePrototype>, PyType *> m_metaclass;

  private:
	PyType(PyType *);

	PyType(TypePrototype &type_prototype);
	PyType(std::unique_ptr<TypePrototype> &&type_prototype);

	static PyResult<PyType *> create(PyType *);

  public:
	static PyType *initialize(TypePrototype &type_prototype);
	static PyType *initialize(std::unique_ptr<TypePrototype> &&type_prototype);

	std::string name() const;

	PyResult<PyObject *> __call__(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __getattribute__(PyObject *attribute) const;

	PyResult<PyObject *> new_(PyTuple *args, PyDict *kwargs) const override;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);

	std::string to_string() const override;

	const TypePrototype &underlying_type() const
	{
		if (std::holds_alternative<std::reference_wrapper<TypePrototype>>(m_underlying_type)) {
			return std::get<std::reference_wrapper<TypePrototype>>(m_underlying_type).get();
		} else {
			return *std::get<std::unique_ptr<TypePrototype>>(m_underlying_type);
		}
	}

	TypePrototype &underlying_type()
	{
		if (std::holds_alternative<std::reference_wrapper<TypePrototype>>(m_underlying_type)) {
			return std::get<std::reference_wrapper<TypePrototype>>(m_underlying_type).get();
		} else {
			return *std::get<std::unique_ptr<TypePrototype>>(m_underlying_type);
		}
	}

	void visit_graph(Visitor &visitor) override;

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();

	PyType *static_type() const override;

	PyResult<PyList *> mro();

	bool issubclass(const PyType *) const;

	bool issubtype(const TypePrototype &other) const;

	std::optional<PyResult<PyObject *>> lookup(PyObject *name) const;

	PyDict *dict() { return m_attributes; }

	static PyResult<PyObject *> heap_object_allocation(PyType *);

	static PyResult<const PyType *> calculate_metaclass(const PyType *type_,
		const std::vector<PyType *> &bases);

  protected:
	PyResult<PyTuple *> mro_internal() const;

  private:
	PyResult<std::monostate> ready();
	PyResult<std::monostate> add_operators();
	PyResult<std::monostate> add_methods();
	PyResult<std::monostate> add_members();
	PyResult<std::monostate> add_properties();
	PyResult<std::monostate> inherit_slots(PyType *base);
	void inherit_special(PyType *base);
	void fixup_slots();

	PyResult<std::monostate> initialize(const std::string &name,
		PyType *base,
		std::vector<PyType *> bases,
		const PyDict *ns);

	static PyResult<PyType *> build_type(const PyType *metatype,
		PyString *type_name,
		PyType *base,
		std::vector<PyType *> bases,
		const PyDict *ns);

	using BasePair = std::tuple<PyType *, std::vector<PyType *>>;
	static PyResult<std::variant<BasePair, PyObject *>> compute_bases(const PyType *type_,
		std::vector<PyType *> bases,
		PyTuple *args,
		PyDict *kwargs);

	static PyResult<PyType *> best_base(const std::vector<PyType *> &bases);
};

}// namespace py
