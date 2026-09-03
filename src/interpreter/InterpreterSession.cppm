module;
#include "core.hpp"


export module py.runtime:interpreter_session;
// :interpreter (not a forward declaration) because m_interpreters holds
// unique_ptr<Interpreter>: remove_if and the destructor erase elements, which
// instantiates default_delete<Interpreter> and needs the complete type.
import :interpreter;
import std;

export class Interpreter;
class BytecodeProgram;

class InterpreterSession
	: NonCopyable
	, NonMoveable
{

	std::list<std::unique_ptr<Interpreter>> m_interpreters;

  public:
	Interpreter &interpreter()
	{
		ASSERT(!m_interpreters.empty());
		return *m_interpreters.back();
	}

	~InterpreterSession();

	const Interpreter &interpreter() const
	{
		ASSERT(!m_interpreters.empty());
		return *m_interpreters.back();
	}

	const std::list<std::unique_ptr<Interpreter>> &interpreters() const { return m_interpreters; }

	void shutdown(Interpreter &interpreter);

	Interpreter &start_new_interpreter(const BytecodeProgram &program);
};
