#include "runtime/PyModule.hpp"

namespace py {

PyModule *builtins_module(Interpreter &interpreter);
PyModule *codecs_module();
PyModule *collections_module();
PyModule *errno_module();
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
}// namespace py
