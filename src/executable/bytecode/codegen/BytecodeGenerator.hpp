#pragma once

#include "ast/AST.hpp"
#include "ast/optimizers/ConstantFolding.hpp"

#include "forward.hpp"
#include "utilities.hpp"

#include "executable/FunctionBlock.hpp"
#include "executable/Label.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"

#include <set>

namespace codegen {

class BytecodeGenerator;

struct FunctionInfo
{
	size_t function_id;
	BytecodeGenerator *generator;

	FunctionInfo(size_t, BytecodeGenerator *);
	~FunctionInfo();
};

class BytecodeGenerator : public ast::CodeGenerator
{
	class ASTContext
	{
		std::stack<std::shared_ptr<ast::Arguments>> m_local_args;
		std::vector<const ast::ASTNode *> m_parent_nodes;

	  public:
		void push_local_args(std::shared_ptr<ast::Arguments> args)
		{
			m_local_args.push(std::move(args));
		}
		void pop_local_args() { m_local_args.pop(); }

		bool has_local_args() const { return !m_local_args.empty(); }

		void push_node(const ast::ASTNode *node) { m_parent_nodes.push_back(node); }
		void pop_node() { m_parent_nodes.pop_back(); }

		const std::shared_ptr<ast::Arguments> &local_args() const { return m_local_args.top(); }
		const std::vector<const ast::ASTNode *> &parent_nodes() const { return m_parent_nodes; }
	};

  public:
	static constexpr size_t start_register = 1;

  private:
	FunctionBlocks m_functions;
	size_t m_function_id{ 0 };

	// a non-owning list of all generated Labels
	std::vector<Label *> m_labels;

	std::vector<size_t> m_frame_register_count;
	Register m_last_register{};
	ASTContext m_ctx;

  public:
	BytecodeGenerator();
	~BytecodeGenerator();

	static std::shared_ptr<Program> compile(std::shared_ptr<ast::ASTNode> node,
		std::vector<std::string> argv,
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

	std::shared_ptr<Label> make_label(const std::string &name, size_t function_id)
	{
		spdlog::debug("New label to be added: name={} function_id={}", name, function_id);
		auto new_label = std::make_shared<Label>(name, function_id);

		ASSERT(std::find(m_labels.begin(), m_labels.end(), new_label.get()) == m_labels.end())

		m_labels.emplace_back(new_label.get());

		return new_label;
	}

	const Label &label(const Label &l) const
	{
		if (auto it = std::find(m_labels.begin(), m_labels.end(), &l); it != m_labels.end()) {
			return **it;
		} else {
			ASSERT_NOT_REACHED()
		}
	}

	const std::vector<Label *> &labels() const { return m_labels; }

	void bind(Label &label)
	{
		ASSERT(std::find(m_labels.begin(), m_labels.end(), &label) != m_labels.end())
		auto &instructions = function(label.function_id());
		const size_t current_instruction_position = instructions.size();
		label.set_position(current_instruction_position);
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
#define __AST_NODE_TYPE(NodeType) void visit(const ast::NodeType *node) override;
	AST_NODE_TYPES
#undef __AST_NODE_TYPE

	template<typename T>
	Register generate(const T *node, size_t function_id) requires std::is_base_of_v<ast::ASTNode, T>
	{
		m_ctx.push_node(node);
		const auto old_function_id = m_function_id;
		m_function_id = function_id;
		node->codegen(this);
		m_function_id = old_function_id;
		m_ctx.pop_node();
		return m_last_register;
	}

	std::shared_ptr<Program> generate_executable(std::string, std::vector<std::string>);
	void relocate_labels(const FunctionBlocks &functions);
};
}// namespace codegen