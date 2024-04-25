#pragma once

#include "PyObject.hpp"
#include "PyTuple.hpp"
#include "runtime/Value.hpp"

#include <optional>

namespace py {

class PyString : public PyBaseObject
{
	friend class ::Heap;
	std::string m_value;

	PyString(PyType *);

  public:
	struct FormatSpec
	{
		size_t start;
		size_t end;
		std::optional<std::string> mapping;
		std::optional<char> conversion_flag;
		std::optional<uint32_t> minimum_width;
		std::optional<uint32_t> precision;
		std::optional<char> conversion_type;

		PyResult<std::string> apply(PyObject *obj) const;
	};

	struct ReplacementField
	{
		enum class Conversion {
			REPR = 1,
			STR = 2,
			ASCII = 3,
		};
		std::optional<std::string> field_name;
		std::optional<Conversion> conversion;
		std::optional<FormatSpec> format_spec;
		size_t start;
		size_t end;
		bool display_expression{ false };

		static std::optional<Conversion> get_conversion(char);
	};

  public:
	static PyResult<PyString *> create(const std::string &value);

	static PyResult<PyString *> create(PyObject *);

	static PyResult<PyString *> create(PyString *self, PyTuple *args, PyDict *kwargs)
	{
		// FIXME with proper error handling
		ASSERT(self)
		ASSERT(!args || (args->size() == 0))
		ASSERT(!kwargs)

		return Ok(self);
	}

	const std::string &value() const { return m_value; }
	std::vector<uint32_t> codepoints() const;
	std::optional<uint32_t> codepoint() const;

	std::string to_string() const override { return m_value; }
	size_t size() const;

	static PyResult<PyObject *> __new__(const PyType *type, PyTuple *args, PyDict *kwargs);
	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __str__() const;
	PyResult<int64_t> __hash__() const;
	PyResult<PyObject *> __eq__(const PyObject *obj) const;
	PyResult<PyObject *> __ne__(const PyObject *obj) const;
	PyResult<PyObject *> __lt__(const PyObject *obj) const;
	PyResult<bool> __bool__() const;
	PyResult<PyObject *> __iter__() const;
	PyResult<PyObject *> __getitem__(PyObject *index);

	PyResult<size_t> __len__() const;
	PyResult<PyObject *> __add__(const PyObject *obj) const;
	PyResult<PyObject *> __mod__(const PyObject *obj) const;

	PyResult<PyObject *> isalpha() const;
	PyResult<PyObject *> isalnum() const;
	PyResult<PyObject *> isascii() const;
	PyResult<PyObject *> isdigit() const;
	PyResult<PyObject *> islower() const;
	PyResult<PyObject *> isupper() const;
	PyResult<PyObject *> isidentifier() const;

	PyResult<PyObject *> capitalize() const;
	PyResult<PyObject *> casefold() const;
	PyResult<PyObject *> find(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> rfind(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> count(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> startswith(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> endswith(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> join(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> lower() const;
	PyResult<PyObject *> upper() const;
	PyResult<PyObject *> rpartition(PyTuple *args, PyDict *kwargs) const;
	PyResult<PyObject *> rstrip(PyTuple *args, PyDict *kwargs) const;

	static PyResult<PyObject *> maketrans(PyTuple *args, PyDict *kwargs);

	PyResult<PyObject *> format(PyTuple *args, PyDict *kwargs) const;

	static PyResult<PyString *> convert_to_ascii(PyObject *obj);
	static PyResult<PyString *> from_encoded_object(const PyObject *obj, const std::string& encoding, const std::string& errors);
	static PyResult<PyString *> decode(std::span<const std::byte>, const std::string& encoding, const std::string& errors);

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;

	PyResult<PyObject *> operator[](int64_t) const;

  private:
	PyString(std::string s);

	size_t get_position_from_slice(int64_t) const;

	PyResult<PyString *> printf(const PyObject *values) const;
};

class PyStringIterator : public PyBaseObject
{
	friend class ::Heap;

	const PyString &m_pystring;
	size_t m_current_index{ 0 };

  public:
	PyStringIterator(const PyString &pystring);

	std::string to_string() const override;

	void visit_graph(Visitor &) override;

	PyResult<PyObject *> __repr__() const;
	PyResult<PyObject *> __next__();

	static std::function<std::unique_ptr<TypePrototype>()> type_factory();
	PyType *static_type() const override;
};

}// namespace py
