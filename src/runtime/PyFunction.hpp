#pragma once

#include "PyObject.hpp"
#include "PyTuple.hpp"
#include "vm/VM.hpp"

class PyCode : public PyBaseObject
{
  public:
	class CodeFlags
	{
	  public:
		enum class Flag {
			OPTIMIZED = 0,
			NEWLOCALS = 1,
			VARARGS = 2,
			VARKEYWORDS = 3,
			NESTED = 4,
			GENERATOR = 5,
		};

	  private:
		std::bitset<6> m_flags;

		CodeFlags() = default;

	  public:
		template<typename... Args>
		requires std::conjunction_v<std::is_same<Flag, Args>...> static CodeFlags create(
			Args... args)
		{
			CodeFlags f;
			(f.m_flags.set(static_cast<uint8_t>(args)), ...);
			return f;
		}
		void set(Flag f) { m_flags.set(static_cast<uint8_t>(f)); }
		void reset(Flag f) { m_flags.reset(static_cast<uint8_t>(f)); }
		bool is_set(Flag f) { return m_flags[static_cast<uint8_t>(f)]; }
	};

	const std::shared_ptr<Function> m_function;
	const size_t m_function_id;
	const size_t m_register_count;
	const std::vector<std::string> m_args;
	const std::vector<Value> m_defaults;
	const size_t m_arg_count;
	CodeFlags m_flags;
	PyModule *m_module;

  public:
	PyCode(std::shared_ptr<Function> function,
		size_t function_id,
		std::vector<std::string> args,
		std::vector<Value> defaults,
		size_t arg_count,
		CodeFlags flags,
		PyModule *m_module);

	PyObject *call(PyTuple *args, PyDict *kwargs);
	const std::vector<std::string> &args() const { return m_args; }
	const std::vector<Value> &defaults() const { return m_defaults; }

	std::string to_string() const override { return fmt::format("PyCode"); }

	size_t register_count() const;
	size_t arg_count() const;
	CodeFlags flags() const;

	const std::shared_ptr<Function> &function() const { return m_function; }

	void visit_graph(Visitor &) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};


class PyFunction : public PyBaseObject
{
	const std::string m_name;
	PyCode *m_code;
	PyDict *m_globals;

  public:
	PyFunction(std::string, PyCode *code, PyDict *globals);

	const PyCode *code() const { return m_code; }

	std::string to_string() const override { return fmt::format("PyFunction"); }

	PyObject *call_with_frame(PyDict *locals, PyTuple *args, PyDict *kwargs) const;

	PyObject *__call__(PyTuple *args, PyDict *kwargs);
	const std::string &function_name() const { return m_name; }

	PyDict *globals() const { return m_globals; }

	PyObject *__repr__() const;
	PyObject *__get__(PyObject *instance, PyObject *owner) const;

	void visit_graph(Visitor &) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};


class PyNativeFunction : public PyBaseObject
{
	friend class Heap;

	std::string m_name;
	std::function<PyObject *(PyTuple *, PyDict *)> m_function;
	std::vector<PyObject *> m_captures;

	PyNativeFunction(std::string &&name, std::function<PyObject *(PyTuple *, PyDict *)> &&function);

	// TODO: fix tracking of lambda captures
	template<typename... Args>
	PyNativeFunction(std::string &&name,
		std::function<PyObject *(PyTuple *, PyDict *)> &&function,
		Args &&... args)
		: PyNativeFunction(std::move(name), std::move(function))
	{
		m_captures = std::vector<PyObject *>{ std::forward<Args>(args)... };
	}

  public:
	template<typename... Args>
	static PyNativeFunction *create(std::string name,
		std::function<PyObject *(PyTuple *, PyDict *)> function,
		Args &&... args)
	{
		return VirtualMachine::the().heap().allocate<PyNativeFunction>(
			std::move(name), std::move(function), std::forward<Args>(args)...);
	}

	PyObject *operator()(PyTuple *args, PyDict *kwargs) { return m_function(args, kwargs); }

	std::string to_string() const override
	{
		return fmt::format("PyNativeFunction {}", static_cast<const void *>(&m_function));
	}

	const std::string &name() const { return m_name; }
	PyObject *__call__(PyTuple *args, PyDict *kwargs);
	PyObject *__repr__() const;

	void visit_graph(Visitor &) override;

	static std::unique_ptr<TypePrototype> register_type();
	PyType *type() const override;
};
