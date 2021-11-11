#pragma once

#include "Interpreter.hpp"

#include "utilities.hpp"

#include <list>

class InterpreterSession
	: NonCopyable
	, NonMoveable
{

	std::list<std::unique_ptr<Interpreter>> m_interpreters;

  public:
	Interpreter &interpreter()
	{
		ASSERT(!m_interpreters.empty())
		return *m_interpreters.back();
	}

	const Interpreter &interpreter() const
	{
		ASSERT(!m_interpreters.empty())
		return *m_interpreters.back();
	}

	const std::list<std::unique_ptr<Interpreter>>& interpreters() const {
		return m_interpreters;
	} 

	void shutdown(Interpreter &interpreter)
	{
		const size_t initial_size = m_interpreters.size();
		m_interpreters.erase(std::remove_if(m_interpreters.begin(),
			m_interpreters.end(),
			[&interpreter](const auto &i) { return &interpreter == i.get(); }));
		ASSERT(initial_size != m_interpreters.size())
	}

	Interpreter &start_new_interpreter(std::shared_ptr<Program>);
};