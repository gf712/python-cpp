#include "Modules.hpp"
#include "runtime/PyBytes.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyType.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/ValueError.hpp"
#include "runtime/types/api.hpp"

#include <filesystem>

namespace fs = std::filesystem;

namespace py {

namespace {
	static PyType *s_stat_result = nullptr;

	class PyStatResult : public PyBaseObject
	{
		friend class ::Heap;

	  public:
		mode_t m_st_mode;
		ino_t m_st_ino;
		dev_t m_st_dev;
		nlink_t m_st_nlink;
		uid_t m_st_uid;
		gid_t m_st_gid;
		off_t m_st_size;
		time_t m_st_atime_sec;
		time_t m_st_mtime_sec;
		time_t m_st_ctime_sec;
		long m_st_atime_nsec;
		long m_st_mtime_nsec;
		long m_st_ctime_nsec;

	  private:
		PyStatResult(PyType *type) : PyBaseObject(type->underlying_type()) {}

		PyStatResult(std::unique_ptr<struct stat> &&stat_)
			: PyBaseObject(s_stat_result->underlying_type())
		{
			m_st_mode = stat_->st_mode;
			m_st_ino = stat_->st_ino;
			m_st_dev = stat_->st_dev;
			m_st_nlink = stat_->st_nlink;
			m_st_uid = stat_->st_uid;
			m_st_gid = stat_->st_gid;
			m_st_size = stat_->st_size;
#if defined(__APPLE__)
			m_st_atime_sec = stat_->st_atimespec.tv_sec;
			m_st_mtime_sec = stat_->st_mtimespec.tv_sec;
			m_st_ctime_sec = stat_->st_ctimespec.tv_sec;
			m_st_atime_nsec = stat_->st_atimespec.tv_nsec;
			m_st_mtime_nsec = stat_->st_mtimespec.tv_nsec;
			m_st_ctime_nsec = stat_->st_ctimespec.tv_nsec;
#elif defined(__linux__)
			m_st_atime_sec = stat_->st_atim.tv_sec;
			m_st_mtime_sec = stat_->st_mtim.tv_sec;
			m_st_ctime_sec = stat_->st_ctim.tv_sec;
			m_st_atime_nsec = stat_->st_atim.tv_nsec;
			m_st_mtime_nsec = stat_->st_mtim.tv_nsec;
			m_st_ctime_nsec = stat_->st_ctim.tv_nsec;
#else
			static_assert(false, "Unsupported platform");
#endif
		}

	  public:
		static PyResult<PyStatResult *> create(std::unique_ptr<struct stat> &&stat_)
		{
			auto *result = VirtualMachine::the().heap().allocate<PyStatResult>(std::move(stat_));
			if (!result) { return Err(memory_error(sizeof(PyStatResult))); }
			return Ok(result);
		}

		std::string to_string() const final
		{
			return fmt::format(
				"os.stat_result(st_mode={}, st_ino={}, std_dev={}, st_nlink={}, st_uid={} "
				"st_gid={}, st_size={}. st_atime={}, st_mtime={}, st_ctime={})",
				m_st_mode,
				m_st_ino,
				m_st_dev,
				m_st_nlink,
				m_st_uid,
				m_st_gid,
				m_st_size,
				m_st_atime_sec,
				m_st_mtime_sec,
				m_st_ctime_sec);
		}

		PyResult<PyObject *> __repr__() const { return PyString::create(to_string()); }

		PyType *static_type() const final
		{
			ASSERT(s_stat_result)
			return s_stat_result;
		}

		static PyType *register_type(PyModule *module)
		{
			if (!s_stat_result) {
				s_stat_result =
					klass<PyStatResult>(module, "stat_result")
						.attribute_readonly("st_mode", &PyStatResult::m_st_mode)
						.attribute_readonly("st_ino", &PyStatResult::m_st_ino)
						.attribute_readonly("st_dev", &PyStatResult::m_st_dev)
						.attribute_readonly("st_nlink", &PyStatResult::m_st_nlink)
						.attribute_readonly("st_uid", &PyStatResult::m_st_uid)
						.attribute_readonly("st_gid", &PyStatResult::m_st_gid)
						.attribute_readonly("st_size", &PyStatResult::m_st_size)
						.attribute_readonly("st_atime", &PyStatResult::m_st_atime_sec)
						.attribute_readonly("st_mtime", &PyStatResult::m_st_mtime_sec)
						.attribute_readonly("st_ctime", &PyStatResult::m_st_ctime_sec)
						.attribute_readonly("st_atime_ns", &PyStatResult::m_st_atime_nsec)
						.attribute_readonly("st_mtime_ns", &PyStatResult::m_st_mtime_nsec)
						.attribute_readonly("st_ctime_ns", &PyStatResult::m_st_ctime_nsec)
						.finalize();
			}
			return s_stat_result;
		}
	};
}// namespace

PyModule *posix_module()
{
	auto *s_posix_module = PyModule::create(PyDict::create().unwrap(),
		PyString::create("posix").unwrap(),
		PyString::create("The posix module!").unwrap())
							   .unwrap();

	(void)PyStatResult::register_type(s_posix_module);

	s_posix_module->add_symbol(PyString::create("getcwd").unwrap(),
		PyNativeFunction::create("getcwd",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				if (args) {
					if (args->size() > 0) {
						return Err(type_error(
							"posix.getcwd() takes no arguments ({} given)", args->size()));
					}
				}
				if (kwargs) {
					if (!kwargs->map().empty()) {
						return Err(type_error("posix.getcwd() takes no keyword arguments"));
					}
				}

				// simpler than using posix glibc implementation directly and having to worry about
				// preallocating buffers. Hopefully it is the same result :)
				const auto cwd = std::filesystem::current_path();
				return PyString::create(cwd.string());
			})
			.unwrap());

	s_posix_module->add_symbol(PyString::create("stat").unwrap(),
		PyNativeFunction::create("stat",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(!kwargs || kwargs->map().empty());

				if (!args) { return Err(type_error("posix.stat() takes one argument (0 given)")); }

				if (args->size() != 1) {
					return Err(
						type_error("posix.stat() takes one argument ({} given)", args->size()));
				}

				const auto path = PyObject::from(args->elements()[0]);
				if (path.is_err()) return path;
				if (!as<PyString>(path.unwrap())) {
					return Err(type_error(
						"expected to be string but got '{}'", path.unwrap()->type()->name()));
				}
				const auto *path_cstr = as<PyString>(path.unwrap())->value().c_str();
				auto stat_ = std::make_unique<struct stat>();
				stat(path_cstr, stat_.get());
				// FIXME: handle errors
				ASSERT(stat_);
				return PyStatResult::create(std::move(stat_));
			})
			.unwrap());

	s_posix_module->add_symbol(PyString::create("listdir").unwrap(),
		PyNativeFunction::create("listdir",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(!kwargs || kwargs->map().empty());

				if (args->size() > 1) {
					return Err(type_error(
						"posix.listdir() takes at most one argument ({} given)", args->size()));
				}

				const auto path = args->size() == 1 ? PyObject::from(args->elements()[0])
													: PyObject::from(String{ "." });
				if (path.is_err()) return path;
				if (!as<PyString>(path.unwrap())) {
					return Err(type_error(
						"expected to be string but got '{}'", path.unwrap()->type()->name()));
				}

				const auto dir_name = as<PyString>(path.unwrap())->value();
				const auto dir = fs::path(dir_name);

				{
					std::error_code ec;
					if (!fs::is_directory(dir, ec)) {
						if (ec) {
							// FIXME: return correct error based on errno
							return Err(value_error(
								"[Errno {}] {}: '{}'", ec.value(), ec.message(), dir_name));
						}
						// FIXME: return NotADirectoryError
						constexpr int errno_ = 20;
						return Err(
							value_error("[Errno {}] Not a directory: '{}'", errno_, dir_name));
					}
				}

				auto result = PyList::create();
				if (result.is_err()) return result;
				for (const auto &d : fs::directory_iterator{ dir }) {
					auto name = PyString::create(d.path().filename());
					if (name.is_err()) return name;
					result.unwrap()->elements().push_back(name.unwrap());
				}
				return result;
			})
			.unwrap());

	s_posix_module->add_symbol(PyString::create("fspath").unwrap(),
		PyNativeFunction::create("fspath",
			[](PyTuple *args, PyDict *kwargs) -> py::PyResult<py::PyObject *> {
				ASSERT(!kwargs || kwargs->map().empty());

				if (!args) {
					return Err(type_error("posix.fspath() takes one argument (0 given)"));
				}

				if (args->size() != 1) {
					return Err(
						type_error("posix.fspath() takes one argument ({} given)", args->size()));
				}

				const auto path = PyObject::from(args->elements()[0]);
				if (path.is_err()) return path;
				if (!as<PyString>(path.unwrap()) && !as<PyBytes>(path.unwrap())) {
					// should check __fspath__ slot if not string or bytes
					return Err(type_error("expected str, bytes or os.PathLike object, not {}",
						path.unwrap()->type()->name()));
				}
				return path;
			})
			.unwrap());

	return s_posix_module;
}
}// namespace py
