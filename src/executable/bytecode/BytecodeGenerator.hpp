#pragma once

#include "forward.hpp"
#include "utilities.hpp"

#include "ast/optimizers/ConstantFolding.hpp"
#include "executable/FunctionBlock.hpp"
#include "executable/Label.hpp"
#include "instructions/Instructions.hpp"

#include <memory>
#include <set>
#include <vector>


class BytecodeGenerator;

struct FunctionInfo
{
	size_t function_id;
	BytecodeGenerator *generator;

	FunctionInfo(size_t, BytecodeGenerator *);
	~FunctionInfo();
};


class BytecodeGenerator : NonCopyable
{
  public:
	static constexpr size_t start_register = 1;

  private:
	FunctionBlocks m_functions;

	std::set<Label> m_labels;
	std::vector<size_t> m_frame_register_count;

  public:
	BytecodeGenerator();
	~BytecodeGenerator();

	static std::shared_ptr<Program> compile(std::shared_ptr<ast::ASTNode> node,
		compiler::OptimizationLevel lvl);

	template<typename OpType, typename... Args> void emit(size_t function_id, Args &&... args)
	{
		m_functions.at(function_id)
			.instructions.push_back(std::make_unique<OpType>(std::forward<Args>(args)...));
	}

	friend std::ostream &operator<<(std::ostream &os, BytecodeGenerator &generator);

	std::string to_string() const;

	const FunctionBlocks &functions() const { return m_functions; }

	const InstructionVector &function(size_t idx) const
	{
		ASSERT(idx < m_functions.size())
		return m_functions.at(idx).instructions;
	}

	Label make_label(const std::string &name, size_t function_id)
	{
		spdlog::debug("New label to be added: name={} function_id={}", name, function_id);
		if (auto result = m_labels.emplace(name, function_id); result.second) {
			return *result.first;
		}
		ASSERT_NOT_REACHED()
	}

	Label label(const Label &l) const
	{
		if (auto result = m_labels.find(l); result != m_labels.end()) { return *result; }
		ASSERT_NOT_REACHED()
	}

	const std::set<Label> labels() const { return m_labels; }

	void bind(Label &label)
	{
		if (auto result = m_labels.find(label); result != m_labels.end()) {
			const size_t current_instruction_position = function(result->function_id()).size();
			result->set_position(current_instruction_position);
			return;
		}
		ASSERT_NOT_REACHED()
	}

	size_t register_count() const { return m_frame_register_count.back(); }

	Register allocate_register()
	{
		spdlog::debug("New register: {}", m_frame_register_count.back());
		return m_frame_register_count.back()++;
	}
	FunctionInfo allocate_function();

	void enter_function() { m_frame_register_count.emplace_back(start_register); }

	void exit_function(size_t function_id);

  private:
	std::shared_ptr<Program> generate_executable(std::string);
	void relocate_labels(const FunctionBlocks &functions);
};
