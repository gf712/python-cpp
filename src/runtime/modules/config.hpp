#pragma once

#include "Modules.hpp"
#include <array>

namespace py {
static constexpr std::array builtin_modules{
	std::tuple<std::string_view, PyModule *(*)()>{ "builtin", nullptr },
	std::tuple<std::string_view, PyModule *(*)()>{ "sys", nullptr },
	std::tuple<std::string_view, PyModule *(*)()>{ "_imp", imp_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "_io", io_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "marshal", marshal_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "posix", posix_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "_thread", thread_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "_weakref", weakref_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "_warnings", warnings_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "itertools", itertools_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "_collections", collections_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "time", time_module },
};

inline bool is_builtin(std::string_view name)
{
	for (const auto &[module_name, _] : builtin_modules) {
		if (name == module_name) { return true; }
	}
	return false;
}
}// namespace py
