#include "runtime/PyModule.hpp"

namespace py {

PyModule *builtins_module(Interpreter &interpreter);
PyModule *sys_module(Interpreter &interpreter);

}// namespace py