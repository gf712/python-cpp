#pragma once

#include "utilities.hpp"

class Function;
class VirtualMachine;

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

	virtual const std::shared_ptr<Function> &function(const std::string&) const = 0;
};
