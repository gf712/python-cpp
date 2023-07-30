#pragma once

#include "VariablesResolver.hpp"
#include "ast/AST.hpp"
#include "ast/optimizers/ConstantFolding.hpp"
#include "executable/FunctionBlock.hpp"
#include "executable/Label.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"

#include "forward.hpp"
#include "utilities.hpp"

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

	virtual bool is_function() const { return false; }
};

class BytecodeStaticValue : public ast::Value
{
	size_t m_index;

  public:
	BytecodeStaticValue(size_t index) : ast::Value("constant"), m_index(index) {}

	size_t get_index() const { return m_index; }

	virtual bool is_function() const { return false; }
};

class BytecodeNameValue : public ast::Value
{
	size_t m_index;

  public:
	BytecodeNameValue(const std::string &name, size_t index) : ast::Value(name), m_index(index) {}

	size_t get_index() const { return m_index; }

	virtual bool is_function() const { return false; }
};

class BytecodeStackValue : public ast::Value
{
	Register m_stack_index;

  public:
	BytecodeStackValue(const std::string &name, Register stack_index)
		: ast::Value(name), m_stack_index(stack_index)
	{}

	Register get_stack_index() const { return m_stack_index; }

	virtual bool is_function() const { return false; }
};

class BytecodeFreeValue : public ast::Value
{
	Register m_free_var_index;

  public:
	BytecodeFreeValue(const std::string &name, Register free_var_index)
		: ast::Value(name), m_free_var_index(free_var_index)
	{}

	Register get_free_var_index() const { return m_free_var_index; }

	virtual bool is_function() const { return false; }
};

class BytecodeFunctionValue : public BytecodeValue
{
	FunctionInfo m_info;

  public:
	BytecodeFunctionValue(const std::string &name, Register register_, FunctionInfo &&info)
		: BytecodeValue(name, register_), m_info(std::move(info))
	{}

	const FunctionInfo &function_info() const { return m_info; }

	bool is_function() const final { return true; }
};

class BytecodeGenerator : public ast::CodeGenerator
{
	friend FunctionInfo;

	class ASTContext
	{
		std::stack<std::shared_ptr<ast::Arguments>> m_local_args;
		std::vector<const ast::ASTNode *> m_parent_nodes;
		std::shared_ptr<Label> m_current_loop_start_label;
		std::shared_ptr<Label> m_current_loop_end_label;

	  public:
		void push_local_args(std::shared_ptr<ast::Arguments> args)
		{
			m_local_args.push(std::move(args));
		}
		void pop_local_args() { m_local_args.pop(); }
		bool has_local_args() const { return !m_local_args.empty(); }
		const std::shared_ptr<ast::Arguments> &local_args() const { return m_local_args.top(); }

		void push_node(const ast::ASTNode *node) { m_parent_nodes.push_back(node); }
		void pop_node() { m_parent_nodes.pop_back(); }
		const std::vector<const ast::ASTNode *> &parent_nodes() const { return m_parent_nodes; }

		std::shared_ptr<Label> set_current_loop_start_label(std::shared_ptr<Label> label)
		{
			m_current_loop_start_label.swap(label);
			return label;
		}

		std::shared_ptr<Label> set_current_loop_end_label(std::shared_ptr<Label> label)
		{
			m_current_loop_end_label.swap(label);
			return label;
		}

		const std::shared_ptr<Label> &get_current_loop_start_label() const
		{
			ASSERT(m_current_loop_start_label)
			return m_current_loop_start_label;
		}

		const std::shared_ptr<Label> &get_current_loop_end_label() const
		{
			// should only be used by ast::Break
			ASSERT(m_current_loop_end_label)
			return m_current_loop_end_label;
		}
	};

	struct Scope
	{
		std::string name;
		std::string mangled_name;

		std::unordered_map<std::string,
			std::variant<BytecodeValue *, BytecodeStackValue *, BytecodeFreeValue *>>
			locals;
	};

	struct ScopedClearExceptionBeforeReturn
	{
		ScopedClearExceptionBeforeReturn(BytecodeGenerator &generator_, size_t function_id_)
			: generator(generator_), function_id(function_id_)
		{
			if (!generator.m_clear_exception_before_return_functions.contains(function_id)) {
				requires_cleanup = true;
			}
			generator.m_clear_exception_before_return_functions.insert(function_id);
		}

		~ScopedClearExceptionBeforeReturn()
		{
			if (requires_cleanup) {
				generator.m_clear_exception_before_return_functions.erase(function_id);
			}
		}

	  private:
		BytecodeGenerator &generator;
		const size_t function_id;
		bool requires_cleanup{ false };
	};

	struct ScopedWithStatement
	{
		ScopedWithStatement(BytecodeGenerator &generator_,
			std::function<void(bool)> return_transform,
			size_t function_id_)
			: generator(generator_), function_id(function_id_)
		{
			generator.m_return_transform[function_id].push_back(std::move(return_transform));
			generator.m_current_exception_depth[function_id]++;
		}

		~ScopedWithStatement()
		{
			ASSERT(generator.m_current_exception_depth[function_id] > 0);

			generator.m_return_transform[function_id].pop_back();
			generator.m_current_exception_depth[function_id]--;
		}

	  protected:
		BytecodeGenerator &generator;
		const size_t function_id;
	};

	struct ScopedTryStatement : ScopedWithStatement
	{
		ScopedTryStatement(BytecodeGenerator &generator_,
			std::function<void(bool)> return_transform,
			size_t function_id_)
			: ScopedWithStatement(generator_, return_transform, function_id_)
		{}
	};

  public:
	static constexpr size_t start_register = 1;
	static constexpr size_t start_stack_index = 0;

  private:
	FunctionBlocks m_functions;
	std::unordered_map<std::string, std::reference_wrapper<FunctionBlocks::FunctionType>>
		m_function_map;

	std::unordered_map<std::string, std::unique_ptr<VariablesResolver::Scope>>
		m_variable_visibility;

	size_t m_function_id{ 0 };
	InstructionVector *m_current_block{ nullptr };

	// a non-owning list of all generated Labels
	std::vector<std::shared_ptr<Label>> m_labels;

	std::vector<size_t> m_frame_register_count;
	std::vector<size_t> m_frame_stack_value_count;
	std::vector<size_t> m_frame_free_var_count;

	std::unordered_map<std::string, std::reference_wrapper<size_t>> m_function_free_var_count;

	size_t m_value_index{ 0 };
	ASTContext m_ctx;
	std::stack<Scope> m_stack;

	std::set<size_t> m_clear_exception_before_return_functions;
	std::unordered_map<size_t, std::vector<std::function<void(bool)>>> m_return_transform;
	std::unordered_map<size_t, size_t> m_current_exception_depth;

  public:
	static std::shared_ptr<Program> compile(std::shared_ptr<ast::ASTNode> node,
		std::vector<std::string> argv,
		compiler::OptimizationLevel lvl);

  private:
	BytecodeGenerator();
	~BytecodeGenerator();

	template<typename OpType, typename... Args> void emit(Args &&...args)
	{
		ASSERT(m_current_block)
		m_current_block->push_back(std::make_unique<OpType>(std::forward<Args>(args)...));
	}

	friend std::ostream &operator<<(std::ostream &os, BytecodeGenerator &generator);

	std::string to_string() const;

	const FunctionBlocks &functions() const { return m_functions; }

	const InstructionVector &function(size_t idx) const
	{
		ASSERT(idx < m_functions.functions.size())
		return std::next(m_functions.functions.begin(), idx)->blocks;
	}

	std::shared_ptr<Label> make_label(const std::string &name, size_t function_id)
	{
		spdlog::debug("New label to be added: name={} function_id={}", name, function_id);
		auto new_label = std::make_shared<Label>(name, function_id);

		ASSERT(std::find(m_labels.begin(), m_labels.end(), new_label) == m_labels.end())

		m_labels.emplace_back(new_label);

		return new_label;
	}

	const std::vector<std::shared_ptr<Label>> &labels() const { return m_labels; }

	void bind(const std::shared_ptr<Label> &label)
	{
		ASSERT(std::find(m_labels.begin(), m_labels.end(), label) != m_labels.end())
		auto &instructions = function(label->function_id());
		const auto instructions_size = instructions.size();
		const size_t current_instruction_position = instructions_size;
		label->set_position(current_instruction_position);
		spdlog::debug("bound label {}", label->name());
	}

	size_t register_count() const { return m_frame_register_count.back(); }
	size_t stack_variable_count() const { return m_frame_stack_value_count.back(); }
	size_t free_variable_count() const { return m_frame_free_var_count.back(); }

	Register allocate_register()
	{
		spdlog::debug("New register: {}", m_frame_register_count.back());
		ASSERT(m_frame_register_count.back() < std::numeric_limits<Register>::max());
		return static_cast<Register>(m_frame_register_count.back()++);
	}

	Register allocate_stack_value()
	{
		spdlog::debug("New stack value: {}", m_frame_stack_value_count.back());
		ASSERT(m_frame_stack_value_count.back() < std::numeric_limits<Register>::max());
		return static_cast<Register>(m_frame_stack_value_count.back()++);
	}

	Register allocate_free_value()
	{
		spdlog::debug("New free value: {}", m_frame_free_var_count.back());
		ASSERT(m_frame_free_var_count.back() < std::numeric_limits<Register>::max());
		return static_cast<Register>(m_frame_free_var_count.back()++);
	}

	BytecodeFunctionValue *create_function(const std::string &);

	InstructionVector *allocate_block(size_t);

	void enter_function()
	{
		m_frame_register_count.emplace_back(start_register);
		m_frame_stack_value_count.emplace_back(start_stack_index);
		m_frame_free_var_count.emplace_back(start_stack_index);
	}

	void exit_function(size_t function_id);

	void store_name(const std::string &, BytecodeValue *);
	BytecodeValue *load_var(const std::string &);
	void delete_var(const std::string &);

	BytecodeStaticValue *load_const(const py::Value &, size_t);
	BytecodeNameValue *load_name(const std::string &, size_t);

	BytecodeValue *build_slice(const ast::Subscript::SliceType &);

	std::tuple<size_t, size_t> move_to_stack(const std::vector<Register> &args);
	BytecodeValue *build_dict(const std::vector<std::optional<Register>> &,
		const std::vector<Register> &);
	BytecodeValue *build_dict_simple(const std::vector<std::optional<Register>> &,
		const std::vector<Register> &);
	BytecodeValue *build_list(const std::vector<Register> &);
	BytecodeValue *build_tuple(const std::vector<Register> &);
	BytecodeValue *build_set(const std::vector<Register> &);
	BytecodeValue *build_string(const std::vector<Register> &);
	void emit_call(Register func, const std::vector<Register> &);
	void make_function(Register,
		const std::string &,
		const std::vector<Register> &,
		const std::vector<Register> &,
		const std::optional<Register> &);

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

	BytecodeStackValue *create_stack_value()
	{
		m_values.push_back(std::make_unique<BytecodeStackValue>(
			"%" + std::to_string(m_value_index++), allocate_stack_value()));
		return static_cast<BytecodeStackValue *>(m_values.back().get());
	}

	BytecodeFreeValue *create_free_value()
	{
		m_values.push_back(std::make_unique<BytecodeFreeValue>(
			"%" + std::to_string(m_value_index++), allocate_free_value()));
		return static_cast<BytecodeFreeValue *>(m_values.back().get());
	}

	BytecodeFreeValue *create_free_value(std::string name)
	{
		m_values.push_back(
			std::make_unique<BytecodeFreeValue>(std::move(name), allocate_free_value()));
		return static_cast<BytecodeFreeValue *>(m_values.back().get());
	}

	std::shared_ptr<Program> generate_executable(std::string, std::vector<std::string>);
	void relocate_labels(const FunctionBlocks &functions);

	void set_insert_point(InstructionVector *block) { m_current_block = block; }

	std::string mangle_namespace(std::stack<BytecodeGenerator::Scope> s) const;

	void create_nested_scope(const std::string &name, const std::string &mangled_name);
	std::tuple<std::vector<std::shared_ptr<Label>>,
		std::vector<std::shared_ptr<Label>>,
		BytecodeValue *>
		visit_comprehension(const std::vector<std::shared_ptr<ast::Comprehension>> &comprehensions,
			std::function<BytecodeValue *()> container_builder);

	template<typename FunctionType> ast::Value *generate_function(const FunctionType *);
};
}// namespace codegen
