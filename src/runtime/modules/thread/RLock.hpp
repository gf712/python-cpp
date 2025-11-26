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
	static PyType *s_rlock = nullptr;
}// namespace

class RLock : public PyBaseObject
{
	friend class ::Heap;

	std::recursive_timed_mutex m_mutex;
	bool m_locked{ false };

	RLock(PyType *type) : PyBaseObject(type) {}

	RLock() : PyBaseObject(s_rlock->underlying_type()) {}

  public:
	static PyResult<RLock *> create()
	{
		auto *result = VirtualMachine::the().heap().allocate<RLock>();
		if (!result) { return Err(memory_error(sizeof(RLock))); }
		return Ok(result);
	}

	PyResult<PyObject *> acquire(PyTuple *args, PyDict *kwargs)
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

	PyResult<PyObject *> release()
	{
		if (!m_locked) { return Err(value_error("release unlocked lock")); }
		m_mutex.unlock();
		m_locked = false;
		return Ok(py_none());
	}

	PyResult<PyObject *> __enter__(PyTuple *args, PyDict *kwargs) { return acquire(args, kwargs); }

	PyResult<PyObject *> __exit__(PyTuple *, PyDict *) { return release(); }

	PyType *static_type() const override
	{
		ASSERT(s_rlock)
		return s_rlock;
	}

	static PyType *register_type(PyModule *module)
	{
		if (!s_rlock) {
			s_rlock = klass<RLock>(module, "RLock")
						 .def("release", &RLock::release)
						 .def("acquire", &RLock::acquire)
						 .def("__enter__", &RLock::__enter__)
						 .def("__exit__", &RLock::__exit__)
						 .finalize();
		}
		return s_rlock;
	}
};
}// namespace py
