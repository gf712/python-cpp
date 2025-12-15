#pragma once

#include "ast/AST.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/codegen/VariablesResolver.hpp"

#include <memory>

namespace mlir {
class MLIRContext;
class ModuleOp;
class OpBuilder;
class Block;
namespace func {
	class FuncOp;
}
}// namespace mlir

namespace codegen {

class Context
{
	struct ContextImpl;
	std::unique_ptr<ContextImpl> m_impl;

	Context(std::unique_ptr<ContextImpl>);

  public:
	~Context();

	mlir::MLIRContext &ctx();
	mlir::OpBuilder &builder();
	mlir::ModuleOp &module();
	std::string_view filename() const;

	static Context create();

	ContextImpl *operator->() { return m_impl.get(); }
};

class SSABuilder;

class MLIRGenerator : ast::CodeGenerator
{
	struct MLIRValue;

	// std::unique_ptr<SSABuilder> m_builder;

  private:
	struct Scope
	{
		std::string name;
		std::string mangled_name;
		std::vector<std::function<void(bool)>> finally_blocks;
		std::stack<mlir::Block *> unhappy_path;
		std::deque<bool> clear_exception_before_return;
	};

	struct ClearExceptionBeforeReturn
	{
		ClearExceptionBeforeReturn(Scope &scope) : scope(scope)
		{
			scope.clear_exception_before_return.push_back(true);
		}

		~ClearExceptionBeforeReturn()
		{
			ASSERT(!scope.clear_exception_before_return.empty());
			scope.clear_exception_before_return.pop_back();
		}

		Scope &scope;
	};

	struct RAIIScope
	{
		Scope &scope;
		MLIRGenerator *this_;
		~RAIIScope()
		{
			ASSERT(!this_->m_scope.empty());
			this_->m_scope.pop_back();
		}
	};

  private:
	std::deque<Scope> m_scope;

	Context &m_context;
	std::vector<std::unique_ptr<MLIRValue>> m_values;
	std::unordered_map<std::string, std::unique_ptr<VariablesResolver::Scope>>
		m_variable_visibility;

  private:
	MLIRGenerator(Context &);

  public:
	static bool compile(std::shared_ptr<ast::Module>, std::vector<std::string> argv, Context &);

  private:
#define __AST_NODE_TYPE(NodeType) ast::Value *visit(const ast::NodeType *node) override;
	AST_NODE_TYPES
#undef __AST_NODE_TYPE

	template<typename... Args> MLIRValue *new_value(Args &&...args);

	void store_name(std::string_view name, MLIRValue *value, const SourceLocation &location);
	MLIRValue *load_name(std::string_view name, const SourceLocation &location);
	void delete_name(std::string_view name, const SourceLocation &location);

	MLIRValue *build_slice(const ast::Subscript::SliceType &sliceNode, const SourceLocation &);
	MLIRValue *build_list(const std::vector<MLIRValue *> &els, const SourceLocation &);
	MLIRValue *build_list(const std::vector<MLIRValue *> &els,
		std::vector<bool> requires_expansion,
		const SourceLocation &);
	MLIRValue *build_tuple(const std::vector<MLIRValue *> &els, const SourceLocation &);
	MLIRValue *build_tuple(const std::vector<MLIRValue *> &els,
		std::vector<bool> requires_expansion,
		const SourceLocation &);
	MLIRValue *build_dict(const std::vector<MLIRValue *> &keys,
		const std::vector<MLIRValue *> &values,
		const SourceLocation &location);
	MLIRValue *build_dict(const std::vector<MLIRValue *> &keys,
		const std::vector<MLIRValue *> &values,
		std::vector<bool> requires_expansion,
		const SourceLocation &location);
	MLIRValue *build_set(std::vector<MLIRValue *> els,
		std::vector<bool> requires_expansion,
		const SourceLocation &);
	MLIRValue *build_set(std::vector<MLIRValue *> els, const SourceLocation &);

	std::optional<mlir::Block *> unhappy_path() const;

	void return_value(MLIRValue *, const SourceLocation &);

	void assign(const std::shared_ptr<ast::ASTNode> &target,
		MLIRValue *src,
		const SourceLocation &source_location);

	MLIRGenerator::MLIRValue *make_function(const std::string &function_name,
		const std::string &mangled_name,
		const std::shared_ptr<ast::Arguments> &args,
		const std::vector<std::shared_ptr<ast::ASTNode>> &body,
		const std::vector<std::shared_ptr<ast::ASTNode>> &decorator_list,
		bool is_anon,
		bool is_async,
		const SourceLocation &source_location);

	MLIRValue *build_comprehension(std::string_view function_name,
		std::function<MLIRValue *()> container_factory,
		std::function<void(MLIRValue *)> container_update,
		const std::vector<std::shared_ptr<ast::Comprehension>> &generators,
		const SourceLocation &source_location);

	[[nodiscard]] RAIIScope setup_function(mlir::func::FuncOp &fn,
		const std::string &name,
		const std::string &mangled_name);

	[[nodiscard]] RAIIScope create_nested_scope(const std::string &name,
		const std::string &mangled_name);

	std::string mangle_namespace(const std::deque<Scope> &s) const;

	Scope &scope() { return m_scope.back(); }
	const Scope &scope() const { return m_scope.back(); }
};
}// namespace codegen