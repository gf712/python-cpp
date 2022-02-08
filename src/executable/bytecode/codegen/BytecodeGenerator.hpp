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
	FunctionBlock &function;
	BytecodeGenerator *generator;

	FunctionInfo(size_t, FunctionBlock &, BytecodeGenerator *);
};

class BytecodeValue : public ast::Value
{
	Register m_register;

  public:
	BytecodeValue(const std::string &name, Register register_)
		: ast::Value(name), m_register(register_)
	{}

	Register get_register() const { return m_register; }
};

class BytecodeFunctionValue : public BytecodeValue
{
	FunctionInfo m_info;

  public:
	BytecodeFunctionValue(const std::string &name, Register register_, FunctionInfo &&info)
		: BytecodeValue(name, register_), m_info(std::move(info))
	{}

	const FunctionInfo &function_info() const { return m_info; }
};

class BytecodeGenerator : public ast::CodeGenerator
{
	class ASTContext
	{
		std::stack<std::shared_ptr<ast::Arguments>> m_local_args;
		std::vector<const ast::ASTNode *> m_parent_nodes;
		std::vector<std::vector<std::string>> m_local_globals;

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

	struct Scope
	{
		std::string name;
		std::vector<std::string> local_globals;
	};

  public:
	static constexpr size_t start_register = 1;

  private:
	FunctionBlocks m_functions;
	size_t m_function_id{ 0 };
	InstructionBlock *m_current_block{ nullptr };

	// a non-owning list of all generated Labels
	std::vector<Label *> m_labels;

	std::vector<size_t> m_frame_register_count;
	size_t m_value_index{ 0 };
	ASTContext m_ctx;
	std::stack<Scope> m_stack;

  public:
	BytecodeGenerator();
	~BytecodeGenerator();

	static std::shared_ptr<Program> compile(std::shared_ptr<ast::ASTNode> node,
		std::vector<std::string> argv,
		compiler::OptimizationLevel lvl);

	template<typename OpType, typename... Args> void emit(Args &&... args)
	{
		ASSERT(m_current_block)
		m_current_block->push_back(std::make_unique<OpType>(std::forward<Args>(args)...));
	}

	friend std::ostream &operator<<(std::ostream &os, BytecodeGenerator &generator);

	std::string to_string() const;

	const FunctionBlocks &functions() const { return m_functions; }

	const std::list<InstructionBlock> &function(size_t idx) const
	{
		ASSERT(idx < m_functions.size())
		return std::next(m_functions.begin(), idx)->blocks;
	}

	const InstructionBlock &function(size_t idx, size_t block) const
	{
		ASSERT(idx < m_functions.size())
		auto f = std::next(m_functions.begin(), idx);
		ASSERT(block < f->blocks.size())
		return *std::next(f->blocks.begin(), block);
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
		auto &blocks = function(label.function_id());
		const auto instructions_size = std::transform_reduce(
			blocks.begin(), blocks.end(), 0u, std::plus<size_t>{}, [](const auto &ins) {
				return ins.size();
			});
		const size_t current_instruction_position = instructions_size;
		label.set_position(current_instruction_position);
	}

	size_t register_count() const { return m_frame_register_count.back(); }

	Register allocate_register()
	{
		spdlog::debug("New register: {}", m_frame_register_count.back());
		return m_frame_register_count.back()++;
	}
	BytecodeFunctionValue *create_function(const std::string &);

	InstructionBlock *allocate_block(size_t);

	void enter_function() { m_frame_register_count.emplace_back(start_register); }

	void exit_function(size_t function_id);

	void store_name(const std::string &, BytecodeValue *);
	BytecodeValue *load_name(const std::string &);

  private:
#define __AST_NODE_TYPE(NodeType) ast::Value *visit(const ast::NodeType *node) override;
	AST_NODE_TYPES
#undef __AST_NODE_TYPE

	BytecodeValue *generate(const ast::ASTNode *node, size_t function_id)
	{
		m_ctx.push_node(node);
		const auto old_function_id = m_function_id;
		m_function_id = function_id;
		auto *value = node->codegen(this);
		m_function_id = old_function_id;
		m_ctx.pop_node();
		return static_cast<BytecodeValue *>(value);
	}

	BytecodeValue *create_value(const std::string &name)
	{
		m_values.push_back(std::make_unique<BytecodeValue>(
			name + std::to_string(m_value_index++), allocate_register()));
		return static_cast<BytecodeValue *>(m_values.back().get());
	}

	BytecodeValue *create_return_value()
	{
		m_values.push_back(
			std::make_unique<BytecodeValue>("%" + std::to_string(m_value_index++), 0));
		return static_cast<BytecodeValue *>(m_values.back().get());
	}

	BytecodeValue *create_value()
	{
		m_values.push_back(std::make_unique<BytecodeValue>(
			"%" + std::to_string(m_value_index++), allocate_register()));
		return static_cast<BytecodeValue *>(m_values.back().get());
	}

	std::shared_ptr<Program> generate_executable(std::string, std::vector<std::string>);
	void relocate_labels(const FunctionBlocks &functions);

	void set_insert_point(InstructionBlock *block) { m_current_block = block; }

	std::string mangle_namespace(std::stack<BytecodeGenerator::Scope> s) const;
};
}// namespace codegen