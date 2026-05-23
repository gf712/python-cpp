#include "Modules.hpp"
#include "memory/GarbageCollector.hpp"
#include "memory/Heap.hpp"
#include "runtime/PyBool.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyNone.hpp"
#include "runtime/PyString.hpp"
#include "vm/VM.hpp"

namespace py {

namespace {

	PyResult<PyObject *> gc_collect(PyTuple *, PyDict *)
	{
		// Force a full mark/sweep regardless of pause flag or cadence
		// counter. CPython's gc.collect() returns the number of objects
		// collected; we don't track that, so we just return 0 for
		// compatibility.
		auto &heap = VirtualMachine::the().heap();
		heap.garbage_collector().force_run(heap);
		return Ok(py_none());
	}

	PyResult<PyObject *> gc_enable(PyTuple *, PyDict *)
	{
		// Idempotent: calling enable() on an already-enabled GC must
		// not abort. pause()/resume() ASSERT on state mismatch, so we
		// gate on is_active() here.
		auto &gc = VirtualMachine::the().heap().garbage_collector();
		if (!gc.is_active()) { gc.resume(); }
		return Ok(py_none());
	}

	PyResult<PyObject *> gc_disable(PyTuple *, PyDict *)
	{
		auto &gc = VirtualMachine::the().heap().garbage_collector();
		if (gc.is_active()) { gc.pause(); }
		return Ok(py_none());
	}

	PyResult<PyObject *> gc_isenabled(PyTuple *, PyDict *)
	{
		auto &gc = VirtualMachine::the().heap().garbage_collector();
		return Ok(gc.is_active() ? py_true() : py_false());
	}

}// namespace

PyModule *gc_module()
{
	auto *m = PyModule::create(PyDict::create().unwrap(),
		PyString::create("gc").unwrap(),
		PyString::create("Garbage collector interface.").unwrap())
				  .unwrap();

	m->add_symbol(PyString::create("collect").unwrap(),
		PyNativeFunction::create("collect", &gc_collect).unwrap());
	m->add_symbol(PyString::create("enable").unwrap(),
		PyNativeFunction::create("enable", &gc_enable).unwrap());
	m->add_symbol(PyString::create("disable").unwrap(),
		PyNativeFunction::create("disable", &gc_disable).unwrap());
	m->add_symbol(PyString::create("isenabled").unwrap(),
		PyNativeFunction::create("isenabled", &gc_isenabled).unwrap());

	return m;
}

}// namespace py
