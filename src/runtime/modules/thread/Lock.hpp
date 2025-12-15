#include "runtime/MemoryError.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyInteger.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyType.hpp"
#include "runtime/TypeError.hpp"
#include "runtime/ValueError.hpp"
#include "runtime/types/api.hpp"
#include "vm/VM.hpp"

namespace py {

namespace {
	static PyType *s_lock = nullptr;
}// namespace

class Lock : public PyBaseObject
{
	friend class ::Heap;

	std::timed_mutex m_mutex;
	bool m_locked{ false };

	Lock(PyType *type) : PyBaseObject(type) {}

	Lock() : PyBaseObject(s_lock->underlying_type()) {}

  public:
	static PyResult<Lock *> create()
	{
		auto *result = VirtualMachine::the().heap().allocate<Lock>();
		if (!result) { return Err(memory_error(sizeof(Lock))); }
		return Ok(result);
	}

	PyResult<PyObject *> acquire_lock(PyTuple *args, PyDict *kwargs)
	{
		PyResult<PyObject *> blocking_ = [args, kwargs]() -> PyResult<PyObject *> {
			if (args) {
				if (args->size() > 0) { return PyObject::from(args->elements()[0]); }
			} else if (kwargs) {
				if (kwargs->map().contains(String{ "blocking" })) {
					return PyObject::from(kwargs->map().at(String{ "blocking" }));
				}
			}
			return Ok(py_true());
		}();

		if (blocking_.is_err()) { return Err(blocking_.unwrap_err()); }

		if (!as<PyBool>(blocking_.unwrap())) {
			return Err(type_error("Expected blocking to be of type bool"));
		}

		const bool blocking = as<PyBool>(blocking_.unwrap()) == py_true();

		PyResult<PyObject *> timeout_ = [args, kwargs]() -> PyResult<PyObject *> {
			if (args) {
				if (args->size() > 1) { return PyObject::from(args->elements()[1]); }
			} else if (kwargs) {
				if (kwargs->map().contains(String{ "timeout" })) {
					return PyObject::from(kwargs->map().at(String{ "timeout" }));
				}
			}
			return Ok(py_none());
		}();

		if (timeout_.is_err()) { return Err(timeout_.unwrap_err()); }

		if (!blocking && timeout_.unwrap() == py_none()) {
			return Err(value_error("can't specify a timeout for a non-blocking call"));
		}

		auto timeout = [timeout_]() -> PyResult<int64_t> {
			if (timeout_.unwrap() == py_none()) {
				// block for 1,000,000 seconds
				return Ok(1'000'000'000'000);
			} else if (!as<PyInteger>(timeout_.unwrap())) {
				return Err(type_error("Expected timeout to be of type integer"));
			} else {
				return Ok(as<PyInteger>(timeout_.unwrap())->as_i64());
			}
		}();

		if (timeout.unwrap() < 0) { return Err(value_error("timeout value must be positive")); }

		m_locked = [blocking, timeout, this]() {
			if (!blocking) {
				return m_mutex.try_lock();
			} else {
				return m_mutex.try_lock_for(std::chrono::microseconds(timeout.unwrap()));
			}
		}();

		return m_locked ? Ok(py_true()) : Ok(py_false());
	}

	PyResult<PyObject *> acquire(PyTuple *args, PyDict *kwargs)
	{
		return acquire_lock(args, kwargs);
	}

	PyResult<PyObject *> release_lock()
	{
		if (!m_locked) { return Err(value_error("release unlocked lock")); }
		m_mutex.unlock();
		m_locked = false;
		return Ok(py_none());
	}

	PyResult<PyObject *> release() { return release_lock(); }

	PyResult<PyObject *> locked_lock() { return m_locked ? Ok(py_true()) : Ok(py_false()); }

	PyResult<PyObject *> locked() { return locked_lock(); }

	PyResult<PyObject *> __enter__(PyTuple *args, PyDict *kwargs)
	{
		return acquire_lock(args, kwargs);
	}

	PyResult<PyObject *> __exit__(PyTuple *, PyDict *) { return release_lock(); }

	PyType *static_type() const override
	{
		ASSERT(s_lock);
		return s_lock;
	}

	static PyType *register_type(PyModule *module)
	{
		if (!s_lock) {
			s_lock = klass<Lock>(module, "lock")
						 .def("locked", &Lock::locked)
						 .def("locked_lock", &Lock::locked_lock)
						 .def("release", &Lock::release)
						 .def("release_lock", &Lock::release_lock)
						 .def("acquire", &Lock::acquire)
						 .def("acquire_lock", &Lock::acquire_lock)
						 .def("__enter__", &Lock::__enter__)
						 .def("__exit__", &Lock::__exit__)
						 .disable_new()
						 .finalize();
		}
		module->add_symbol(PyString::create("LockType").unwrap(), s_lock);
		return s_lock;
	}
};
}// namespace py
