#pragma once

#include "forward.hpp"
#include <vector>
#include <memory>

class Bytecode
{
	std::vector<std::unique_ptr<Instruction>> m_instructions;
	std::vector<std::pair<size_t, std::string>> m_functions;
	size_t m_virtual_register_count;

  public:
	Bytecode(std::vector<std::unique_ptr<Instruction>> &&ins,
		std::vector<std::pair<size_t, std::string>> &&funcs,
		size_t virtual_register_count);

	size_t start_offset() const { return m_functions.back().first; }
	size_t virtual_register_count() const { return m_virtual_register_count; }

	const std::vector<std::unique_ptr<Instruction>> &instructions() const { return m_instructions; }
	const std::vector<std::pair<size_t, std::string>> &functions() const { return m_functions; }

	size_t function_offset(size_t function_id) const
	{
		return m_functions[function_id - 1].first;
	}

	std::string to_string() const;
};


class BytecodeGenerator
{
  public:
	static constexpr size_t start_register = 1;

  private:
	std::vector<std::vector<std::unique_ptr<Instruction>>> m_functions;
	Register m_register_index{ start_register };

  public:
	BytecodeGenerator();
	~BytecodeGenerator();

	static std::shared_ptr<Bytecode> compile(std::shared_ptr<ast::ASTNode> node);

	template<typename OpType, typename... Args> void emit(size_t function_id, Args &&... args)
	{
		m_functions.at(function_id)
			.push_back(std::make_unique<OpType>(std::forward<Args>(args)...));
	}

	friend std::ostream &operator<<(std::ostream &os, BytecodeGenerator &generator);

	std::string to_string() const;

	const std::vector<std::unique_ptr<Instruction>> &function(size_t idx) const
	{
		return m_functions.at(idx);
	}


	size_t register_count() const { return m_register_index; }

	Register allocate_register() { return m_register_index++; }
	size_t allocate_function();

  private:
	std::shared_ptr<Bytecode> generate_executable();
};
