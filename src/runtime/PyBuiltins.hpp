#include "PyModule.hpp"
#include <memory>

std::shared_ptr<PyModule> fetch_builtins(Interpreter& interpreter);