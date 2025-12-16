#include "../Modules.hpp"
#include "runtime/PyArgParser.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyFunction.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "runtime/Value.hpp"
#include "runtime/ValueError.hpp"

#include <csignal>
#include <cstdint>
#include <limits>
#include <variant>

namespace py {

namespace {
	static constexpr std::string_view kDoc =
		R"(This module provides mechanisms to use signal handlers in Python.

Functions:

alarm() -- cause SIGALRM after a specified time [Unix only]
setitimer() -- cause a signal (described below) after a specified
               float time and the timer may restart then [Unix only]
getitimer() -- get current value of timer [Unix only]
signal() -- set the action for a given signal
getsignal() -- get the signal action for a given signal
pause() -- wait until a signal arrives [Unix only]
default_int_handler() -- default SIGINT handler

signal constants:
SIG_DFL -- used to refer to the system default handler
SIG_IGN -- used to ignore the signal
NSIG -- number of defined signals
SIGINT, SIGTERM, etc. -- signal numbers

itimer constants:
ITIMER_REAL -- decrements in real time, and delivers SIGALRM upon
               expiration
ITIMER_VIRTUAL -- decrements only when the process is executing,
               and delivers SIGVTALRM upon expiration
ITIMER_PROF -- decrements both when the process is executing and
               when the system is executing on behalf of the process.
               Coupled with ITIMER_VIRTUAL, this timer is usually
               used to profile the time spent by the application
               in user and kernel space. SIGPROF is delivered upon
               expiration.


*** IMPORTANT NOTICE ***
A signal handler function is called with two arguments:
the first is the signal number, the second is the interrupted stack frame.)";
}

static PyDict *handlers = nullptr;

PyResult<PyObject *> signal(PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<int64_t, PyObject *>::unpack_tuple(args,
		kwargs,
		"signal",
		std::integral_constant<size_t, 2>{},
		std::integral_constant<size_t, 2>{});

	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [signalnum, handler] = result.unwrap();

	__sighandler_t sighandler = +[](int signumber) {
		if (auto it = handlers->map().find(Number{ signumber }); it != handlers->map().end()) {
			ASSERT(std::holds_alternative<PyObject *>(it->second));
			std::get<PyObject *>(it->second)
				->call(PyTuple::create(Number{ signumber }, py_none()).unwrap(), nullptr);
		}
	};

	PyObject *previous_handler = py_none();

	if (auto it = handlers->map().find(Number{ signalnum }); it != handlers->map().end()) {
		ASSERT(std::holds_alternative<PyObject *>(it->second));
		previous_handler = std::get<PyObject *>(it->second);
	}

	if (signalnum >= NSIG) { return Err(value_error("signal number out of range")); }

	static_assert(NSIG < std::numeric_limits<int>::max());
	if (std::signal(static_cast<int>(signalnum), sighandler) == SIG_ERR) {
		return Err(value_error("error setting signal handler"));
	}

	handlers->insert(Number{ signalnum }, handler);

	return Ok(previous_handler);
}

PyResult<PyObject *> getsignal(PyTuple *args, PyDict *kwargs)
{
	auto result = PyArgsParser<int64_t>::unpack_tuple(args,
		kwargs,
		"getsignal",
		std::integral_constant<size_t, 1>{},
		std::integral_constant<size_t, 1>{});

	if (result.is_err()) { return Err(result.unwrap_err()); }

	auto [signalnum] = result.unwrap();

	if (auto it = handlers->map().find(Number{ signalnum }); it != handlers->map().end()) {
		return PyObject::from(it->second);
	}

	return Ok(py_none());
}

PyModule *signal_module()
{
	auto symbols = PyDict::create().unwrap();
	auto name = PyString::create("_signal").unwrap();
	auto doc = PyString::create(std::string{ kDoc }).unwrap();

	auto *module = PyModule::create(symbols, name, doc).unwrap();

	handlers = PyDict::create().unwrap();
	module->set_context(handlers);

	module->add_symbol(
		PyString::create("signal").unwrap(), PyNativeFunction::create("signal", &signal).unwrap());
	module->add_symbol(PyString::create("getsignal").unwrap(),
		PyNativeFunction::create("getsignal", &getsignal).unwrap());

	return module;
}
}// namespace py
