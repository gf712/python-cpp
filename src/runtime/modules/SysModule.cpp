#include "runtime/MemoryError.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyNamespace.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyType.hpp"
#include "runtime/types/api.hpp"

#include "config.hpp"
#include "interpreter/Interpreter.hpp"
#include "vm/VM.hpp"

#include <filesystem>

using namespace py;

static PyModule *s_sys_module = nullptr;

namespace {
PyResult<PyList *> create_sys_paths(Interpreter &interpreter)
{
	const auto &entry_script = interpreter.entry_script();
	auto entry_parent = PyString::create(std::filesystem::path(entry_script).parent_path());
	if (entry_parent.is_err()) return Err(entry_parent.unwrap_err());
	auto path_list = PyList::create({ entry_parent.unwrap() });

	return path_list;
}

PyResult<PyList *> create_sys_argv(Interpreter &interpreter)
{
	auto argv_list = PyList::create();
	if (argv_list.is_err()) return argv_list;
	for (const auto &arg : interpreter.argv()) {
		auto arg_str = PyString::create(arg);
		if (arg_str.is_err()) { return Err(arg_str.unwrap_err()); }
		argv_list.unwrap()->elements().push_back(arg_str.unwrap());
	}

	return argv_list;
}

PyResult<PyTuple *> builtin_module_names()
{
	std::vector<Value> module_names;
	module_names.reserve(builtin_modules.size());
	for (const auto &[name, _] : builtin_modules) {
		auto name_str = PyString::create(std::string{ name });
		if (name_str.is_err()) return Err(name_str.unwrap_err());
		module_names.push_back(name_str.unwrap());
	}
	return PyTuple::create(module_names);
}

constexpr std::string_view platform()
{
#if defined(__linux__)
	return "linux";
#else
	static_assert(false, "Unsupported platform");
#endif
}

static PyType *s_sys_flags = nullptr;
static PyType *s_sys_version = nullptr;

class Flags : public PyBaseObject
{
	friend class ::Heap;

  public:
	uint8_t m_debug;
	uint8_t m_inspect;
	uint8_t m_interactive;
	uint8_t m_optimize;
	uint8_t m_dont_write_bytecode;
	uint8_t m_no_user_site;
	uint8_t m_no_site;
	uint8_t m_ignore_environment;
	uint8_t m_verbose;
	uint8_t m_bytes_warning;
	uint8_t m_quiet;
	uint8_t m_hash_randomization;
	uint8_t m_isolated;
	bool m_dev_mode;
	uint8_t m_utf8_mode;

  private:
	Flags(uint8_t debug,
		uint8_t inspect,
		uint8_t interactive,
		uint8_t optimize,
		uint8_t dont_write_bytecode,
		uint8_t no_user_site,
		uint8_t no_site,
		uint8_t ignore_environment,
		uint8_t verbose,
		uint8_t bytes_warning,
		uint8_t quiet,
		uint8_t hash_randomization,
		uint8_t isolated,
		bool dev_mode,
		uint8_t utf8_mode)
		: PyBaseObject(s_sys_flags->underlying_type()), m_debug(debug), m_inspect(inspect),
		  m_interactive(interactive), m_optimize(optimize),
		  m_dont_write_bytecode(dont_write_bytecode), m_no_user_site(no_user_site),
		  m_no_site(no_site), m_ignore_environment(ignore_environment), m_verbose(verbose),
		  m_bytes_warning(bytes_warning), m_quiet(quiet), m_hash_randomization(hash_randomization),
		  m_isolated(isolated), m_dev_mode(dev_mode), m_utf8_mode(utf8_mode)
	{}

  public:
	static PyResult<Flags *> create(uint8_t debug,
		uint8_t inspect,
		uint8_t interactive,
		uint8_t optimize,
		uint8_t dont_write_bytecode,
		uint8_t no_user_site,
		uint8_t no_site,
		uint8_t ignore_environment,
		uint8_t verbose,
		uint8_t bytes_warning,
		uint8_t quiet,
		uint8_t hash_randomization,
		uint8_t isolated,
		bool dev_mode,
		uint8_t utf8_mode)
	{
		auto *result = VirtualMachine::the().heap().allocate<Flags>(debug,
			inspect,
			interactive,
			optimize,
			dont_write_bytecode,
			no_user_site,
			no_site,
			ignore_environment,
			verbose,
			bytes_warning,
			quiet,
			hash_randomization,
			isolated,
			dev_mode,
			utf8_mode);
		if (!result) { return Err(memory_error(sizeof(Flags))); }
		return Ok(result);
	}

	std::string to_string() const final
	{
		return fmt::format(
			"sys.flags(debug={}, inspect={}, interactive={}, optimize={}, dont_write_bytecode={}, "
			"no_user_site={}, no_site={}, ignore_environment={}, verbose={}, bytes_warning={}, "
			"quiet={}, hash_randomization={}, isolated={}, dev_mode={}, utf8_mode={})",
			m_debug,
			m_inspect,
			m_interactive,
			m_optimize,
			m_dont_write_bytecode,
			m_no_user_site,
			m_no_site,
			m_ignore_environment,
			m_verbose,
			m_bytes_warning,
			m_quiet,
			m_hash_randomization,
			m_isolated,
			m_dev_mode ? "True" : "False",
			m_utf8_mode);
	}

	PyResult<PyObject *> __repr__() const { return PyString::create(to_string()); }

	PyType *type() const final
	{
		ASSERT(s_sys_flags)
		return s_sys_flags;
	}

	static PyType *register_type(PyModule *module)
	{
		if (!s_sys_flags) {
			s_sys_flags =
				klass<Flags>(module, "flags_type")
					.attribute_readonly("debug", &Flags::m_debug)
					.attribute_readonly("inspect", &Flags::m_inspect)
					.attribute_readonly("interactive", &Flags::m_interactive)
					.attribute_readonly("optimize", &Flags::m_optimize)
					.attribute_readonly("dont_write_bytecode", &Flags::m_dont_write_bytecode)
					.attribute_readonly("no_user_site", &Flags::m_no_user_site)
					.attribute_readonly("no_site", &Flags::m_no_site)
					.attribute_readonly("ignore_environment", &Flags::m_ignore_environment)
					.attribute_readonly("verbose", &Flags::m_verbose)
					.attribute_readonly("bytes_warning", &Flags::m_bytes_warning)
					.attribute_readonly("quiet", &Flags::m_quiet)
					.attribute_readonly("hash_randomization", &Flags::m_hash_randomization)
					.attribute_readonly("isolated", &Flags::m_isolated)
					.attribute_readonly("dev_mode", &Flags::m_dev_mode)
					.attribute_readonly("utf8_mode", &Flags::m_utf8_mode)
					.finalize();
		}
		return s_sys_flags;
	}
};

class Version : public PyBaseObject
{
	friend class ::Heap;

  public:
	uint8_t m_major;
	uint8_t m_minor;
	uint8_t m_micro;
	std::string m_release_level;
	uint8_t m_serial;

  private:
	Version(uint8_t major, uint8_t minor, uint8_t micro, std::string release_level, uint8_t serial)
		: PyBaseObject(s_sys_version->underlying_type()), m_major(major), m_minor(minor),
		  m_micro(micro), m_release_level(release_level), m_serial(serial)
	{}

  public:
	static PyResult<Version *> create(uint8_t major,
		uint8_t minor,
		uint8_t micro,
		std::string release_level,
		uint8_t serial)
	{
		auto *result = VirtualMachine::the().heap().allocate<Version>(
			major, minor, micro, release_level, serial);
		if (!result) { return Err(memory_error(sizeof(Version))); }
		return Ok(result);
	}

	std::string to_string() const final
	{
		return fmt::format(
			"sys.version_info(major={}, minor={}, micro={}, releaselevel={}, serial={})",
			m_major,
			m_minor,
			m_micro,
			m_release_level,
			m_serial);
	}

	PyResult<PyObject *> __repr__() const { return PyString::create(to_string()); }

	PyType *type() const final
	{
		ASSERT(s_sys_version)
		return s_sys_version;
	}

	static PyType *register_type(PyModule *module)
	{
		if (!s_sys_version) {
			s_sys_version = klass<Version>(module, "version_info_")
								.attribute_readonly("major", &Version::m_major)
								.attribute_readonly("minor", &Version::m_minor)
								.attribute_readonly("micro", &Version::m_micro)
								.attribute_readonly("releaselevel", &Version::m_release_level)
								.attribute_readonly("serial", &Version::m_serial)
								.finalize();
		}
		return s_sys_version;
	}
};

}// namespace

namespace py {

PyModule *sys_module(Interpreter &interpreter)
{
	auto &heap = VirtualMachine::the().heap();

	if (s_sys_module && heap.slab().has_address(bit_cast<uint8_t *>(s_sys_module))) {
		return s_sys_module;
	}


	s_sys_module = PyModule::create(
		PyDict::create().unwrap(), PyString::create("sys").unwrap(), PyString::create("").unwrap())
					   .unwrap();
	// types
	(void)Version::register_type(s_sys_module);
	(void)Flags::register_type(s_sys_module);

	// symbols
	s_sys_module->add_symbol(
		PyString::create("path").unwrap(), create_sys_paths(interpreter).unwrap());
	s_sys_module->add_symbol(
		PyString::create("argv").unwrap(), create_sys_argv(interpreter).unwrap());

	s_sys_module->add_symbol(
		PyString::create("builtin_module_names").unwrap(), builtin_module_names().unwrap());

	s_sys_module->add_symbol(PyString::create("meta_path").unwrap(), PyList::create().unwrap());

	s_sys_module->add_symbol(PyString::create("platform").unwrap(),
		PyString::create(std::string{ platform() }).unwrap());

	ASSERT(interpreter.modules())
	s_sys_module->add_symbol(PyString::create("modules").unwrap(), interpreter.modules());

	s_sys_module->add_symbol(PyString::create("path_hooks").unwrap(), PyList::create().unwrap());

	s_sys_module->add_symbol(
		PyString::create("path_importer_cache").unwrap(), PyDict::create().unwrap());

	s_sys_module->add_symbol(PyString::create("pycache_prefix").unwrap(), py_none());

	s_sys_module->add_symbol(PyString::create("dont_write_bytecode").unwrap(), py_true());

	// avoid GC'ing implementation and version
	[[maybe_unused]] auto scope = VirtualMachine::the().heap().scoped_gc_pause();

	auto *implementation = PyDict::create().unwrap();
	auto *version = Version::create(3, 19, 0, "prerelease", 0).unwrap();
	auto *flags = Flags::create(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, false, 0).unwrap();

	implementation->insert(String{ "name" }, String{ "python-cpp" });
	// TODO: Add caching and add cache tag.
	// This is used by _bootstrap_external.cache_from_source.
	// The expectation is that if cache_tag is None, we will not try to load the cached path
	implementation->insert(String{ "cache_tag" }, py_none());
	implementation->insert(String{ "version" }, version);
	implementation->insert(String{ "hexversion" }, Number{ 0x03090000 });
	implementation->insert(String{ "_multiarch" }, String{ "x86_64-linux-gnu" });

	s_sys_module->add_symbol(
		PyString::create("implementation").unwrap(), PyNamespace::create(implementation).unwrap());

	s_sys_module->add_symbol(PyString::create("version_info").unwrap(), version);

	s_sys_module->add_symbol(PyString::create("flags").unwrap(), flags);

	return s_sys_module;
}

}// namespace py