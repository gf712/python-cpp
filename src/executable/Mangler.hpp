#pragma once

#include <string>

struct SourceLocation;

class Mangler
{
  public:
	virtual std::string function_mangle(const std::string &module,
		const std::string &function_name,
		const SourceLocation &source_location) const = 0;

	virtual std::string class_mangle(const std::string &module,
		const std::string &class_name,
		const SourceLocation &source_location) const = 0;

	virtual std::string function_demangle(const std::string &mangled_name) const = 0;

	virtual std::string class_demangle(const std::string &mangled_name) const = 0;

	static Mangler &default_mangler();
};