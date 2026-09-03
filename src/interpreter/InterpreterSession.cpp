module;
#include "core.hpp"

module py.runtime;

void InterpreterSession::shutdown(Interpreter &interpreter)
{
	const std::size_t initial_size = m_interpreters.size();
	m_interpreters.remove_if([&interpreter](const auto &i) { return &interpreter == i.get(); });
	ASSERT(initial_size != m_interpreters.size());
}

Interpreter &InterpreterSession::start_new_interpreter(const BytecodeProgram &)
{
	TODO();
	// if (m_interpreters.empty()) {
	// 	auto &interpreter = m_interpreters.emplace_back(std::make_unique<Interpreter>());
	// 	interpreter->setup_main_interpreter(program);
	// } else {
	// 	auto &interpreter = m_interpreters.emplace_back(std::make_unique<Interpreter>());
	// 	interpreter->setup(program);
	// }

	// return *m_interpreters.back();
}
