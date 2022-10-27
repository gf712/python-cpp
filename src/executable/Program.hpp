#pragma once

#include "forward.hpp"
#include "memory/GarbageCollector.hpp"
#include "runtime/forward.hpp"
#include "utilities.hpp"

#include <bitset>

class Function;
class VirtualMachine;

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
		COROUTINE = 6,
	};

  private:
	std::bitset<7> m_flags;

	CodeFlags() = default;

  public:
	template<typename... Args>
	// requires std::conjunction_v<std::is_same<Flag, Args>...>
	static CodeFlags create(Args... args)
	{
		CodeFlags f;
		(f.m_flags.set(static_cast<uint8_t>(args)), ...);
		return f;
	}

	static CodeFlags from_byte(uint8_t b)
	{
		auto f = CodeFlags();
		f.m_flags = std::bitset<7>(b);
		return f;
	}

	void set(Flag f) { m_flags.set(static_cast<uint8_t>(f)); }
	void reset(Flag f) { m_flags.reset(static_cast<uint8_t>(f)); }
	bool is_set(Flag f) const { return m_flags[static_cast<uint8_t>(f)]; }
	std::bitset<7> bits() const { return m_flags; }
};

class Program
	: NonCopyable
	, public std::enable_shared_from_this<Program>
{
	std::string m_filename;
	std::vector<std::string> m_argv;

  protected:
	Program() {}

  public:
	Program(std::string &&filename, std::vector<std::string> &&argv);
	virtual ~Program() {}

	virtual int execute(VirtualMachine *) = 0;

	const std::string &filename() const { return m_filename; }
	const std::vector<std::string> &argv() const { return m_argv; }

	void set_filename(std::string filename) { m_filename = std::move(filename); }

	virtual std::string to_string() const = 0;

	virtual py::PyObject *as_pyfunction(const std::string &function_name,
		const std::vector<py::Value> &default_values,
		const std::vector<py::Value> &kw_default_values,
		py::PyTuple *closure) const = 0;

	virtual py::PyObject *main_function() = 0;

	virtual void visit_functions(Cell::Visitor &) const = 0;

	virtual std::vector<uint8_t> serialize() const = 0;
};
