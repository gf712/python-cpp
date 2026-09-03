export module py.runtime:modules;

import std;

export class Interpreter;// global scope, like its definition in :interpreter

export namespace py {

class PyModule;

PyModule *builtins_module(Interpreter &interpreter);
PyModule *codecs_module();
PyModule *collections_module();
PyModule *errno_module();
PyModule *gc_module();
PyModule *imp_module();
PyModule *io_module();
PyModule *math_module();
PyModule *marshal_module();
PyModule *posix_module();
PyModule *thread_module();
PyModule *weakref_module();
PyModule *warnings_module();
PyModule *itertools_module();
PyModule *signal_module();
PyModule *sre_module();
PyModule *struct_module();
PyModule *sys_module(Interpreter &interpreter);
PyModule *time_module();

constexpr std::array builtin_modules{
	std::tuple<std::string_view, PyModule *(*)()>{ "builtin", nullptr },
	std::tuple<std::string_view, PyModule *(*)()>{ "sys", nullptr },
	std::tuple<std::string_view, PyModule *(*)()>{ "_codecs", codecs_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "_imp", imp_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "_io", io_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "math", math_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "marshal", marshal_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "posix", posix_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "_thread", thread_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "_weakref", weakref_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "_warnings", warnings_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "itertools", itertools_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "_sre", sre_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "_collections", collections_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "time", time_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "_signal", signal_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "errno", errno_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "_struct", struct_module },
	std::tuple<std::string_view, PyModule *(*)()>{ "gc", gc_module },
};

inline bool is_builtin(std::string_view name)
{
	for (const auto &[module_name, _] : builtin_modules) {
		if (name == module_name) { return true; }
	}
	return false;
}
}// namespace py
