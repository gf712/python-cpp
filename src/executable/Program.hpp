#pragma once

#include "utilities.hpp"

class Function;
class VirtualMachine;

namespace py {
class PyObject;
}

class Program : NonCopyable
{
	std::string m_filename;
	std::vector<std::string> m_argv;

  public:
	Program() = delete;
	Program(std::string &&filename, std::vector<std::string> &&argv);
	virtual ~Program() {}

	virtual int execute(VirtualMachine *) = 0;

	const std::string &filename() const { return m_filename; }
	const std::vector<std::string> &argv() const { return m_argv; }

	void set_filename(std::string filename) { m_filename = std::move(filename); }

	virtual std::string to_string() const = 0;

	virtual py::PyObject *as_pyfunction(const std::string &function_name,
		const std::vector<std::string> &argnames,
		const std::vector<py::Value> &default_values,
		const std::vector<py::Value> &kw_default_values,
		size_t positional_args_count,
		size_t kwonly_args_count,
		const py::PyCode::CodeFlags &flags) const = 0;
};
