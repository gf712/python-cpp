#include "runtime/PyModule.hpp"

namespace py {

PyModule *builtins_module(Interpreter &interpreter);
PyModule *imp_module();
PyModule *io_module();
PyModule *marshal_module();
PyModule *posix_module();
PyModule *thread_module();
PyModule *weakref_module();
PyModule *warnings_module();
PyModule *sys_module(Interpreter &interpreter);
}// namespace py