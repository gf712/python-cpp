#include "memory/allocate.hpp"

import py.runtime;

namespace py::detail {

unsigned char *allocate_raw(size_type size, size_type extra_bytes)
{
	return VirtualMachine::the().heap().allocate(size, extra_bytes);
}

}// namespace py::detail
