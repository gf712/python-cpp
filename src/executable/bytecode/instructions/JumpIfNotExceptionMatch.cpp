#include "JumpIfNotExceptionMatch.hpp"
#include "interpreter/Interpreter.hpp"

namespace {
bool has_active_exception(Interpreter &interpreter)
{
	return interpreter.execution_frame()->exception()
		   || interpreter.status() == Interpreter::Status::EXCEPTION;
}
}// namespace

void JumpIfNotExceptionMatch::execute(VirtualMachine &vm, Interpreter &interpreter) const
{
	ASSERT(has_active_exception(interpreter))
	const auto &exception_type = vm.reg(m_exception_type_reg);
	ASSERT(std::holds_alternative<PyObject *>(exception_type))
    auto *exception_type_obj = std::get<PyObject*>(exception_type);

    if (!interpreter.execution_frame()->exception()) {
        // TODO: this is a deprecated API, need to remove usage
        TODO();
    }

    // FIXME: currently all exceptions are singletons so we can compare pointers
    //        However, we should be checking types and subclasses
    if (exception_type_obj == interpreter.execution_frame()->exception()) {
        // skip exception handler body block
        vm.jump_blocks(2);
    }
}