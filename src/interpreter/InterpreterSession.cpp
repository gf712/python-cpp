#include "InterpreterSession.hpp"


Interpreter &InterpreterSession::start_new_interpreter(std::shared_ptr<Program> program)
{
	if (m_interpreters.empty()) {
		auto &interpreter = m_interpreters.emplace_back(std::make_unique<Interpreter>());
		interpreter->setup_main_interpreter(program);
	} else {
		auto &interpreter = m_interpreters.emplace_back(std::make_unique<Interpreter>());
		interpreter->setup(program);
	}

	return *m_interpreters.back();
}
