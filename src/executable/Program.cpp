#include "Program.hpp"


Program::Program(std::string &&filename, std::vector<std::string> &&argv)
	: m_filename(std::move(filename)), m_argv(std::move(argv))
{}