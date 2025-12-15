#include "Mangler.hpp"

#include "ast/AST.hpp"

#include <string_view>

class DefaultMangler : public Mangler
{
  public:
	std::string function_mangle(const std::string &module,
		const std::string &function_name,
		const SourceLocation &source_location) const override
	{
		return fmt::format("{}.{}.{}:{}",
			module,
			function_name,
			source_location.start.row,
			source_location.start.column);
	}

	std::string class_mangle(const std::string &module,
		const std::string &class_name,
		const SourceLocation &source_location) const override
	{
		return fmt::format("{}.__class__{}__.{}:{}",
			module,
			class_name,
			source_location.start.row,
			source_location.start.column);
	}

	std::string function_demangle(const std::string &mangled_name) const override
	{
		auto it = std::find(mangled_name.rbegin(), mangled_name.rend(), '.');
		ASSERT(it != mangled_name.rend());

		const auto end = mangled_name.size() - std::distance(mangled_name.rbegin(), it) - 1;

		std::string_view module_function{ mangled_name.c_str(), end };

		auto mf_it = std::find(module_function.rbegin(), module_function.rend(), '.');
		ASSERT(mf_it != module_function.rend());

		const auto start = module_function.size() - std::distance(module_function.rbegin(), mf_it);

		return std::string{ mangled_name.begin() + start, mangled_name.begin() + end };
	}

	std::string class_demangle(const std::string &mangled_name) const override
	{
		auto it = std::find(mangled_name.rbegin(), mangled_name.rend(), '.');
		ASSERT(it != mangled_name.rend());

		const auto end = mangled_name.size() - std::distance(mangled_name.rbegin(), it) - 1;

		std::string_view module_function{ mangled_name.c_str(), end };

		auto mf_it = std::find(module_function.rbegin(), module_function.rend(), '.');
		ASSERT(mf_it != module_function.rend());

		const auto start = module_function.size() - std::distance(module_function.rbegin(), mf_it);

		std::string_view mangled_class{ mangled_name.c_str() + start, end - start };

		ASSERT(mangled_class.starts_with("__class__"));
		ASSERT(mangled_class.ends_with("__"));

		// extract name from __class__{}__
		return std::string{ mangled_class.begin() + 9, mangled_class.end() - 2 };
	}
};

Mangler &Mangler::default_mangler()
{
	static DefaultMangler m;
	return m;
}
