#include "ast/AST.hpp"
#include "ast/optimizers/ConstantFolding.hpp"

#include "executable/Function.hpp"

namespace llvm {
class Function;
class Type;
class Value;
}// namespace llvm

namespace codegen {
class LLVMGenerator : public ast::CodeGenerator
{
	class LLVMValue;

	struct Context;

	std::unique_ptr<Context> m_ctx;

	LLVMGenerator();

  public:
	static std::shared_ptr<Program> compile(std::shared_ptr<ast::ASTNode> node,
		std::vector<std::string> argv,
		compiler::OptimizationLevel lvl);

  private:
	LLVMValue *generate(const ast::ASTNode *node);

	ast::Value *create_value(llvm::Value *);

#define __AST_NODE_TYPE(NodeType) ast::Value *visit(const ast::NodeType *node) override;
	AST_NODE_TYPES
#undef __AST_NODE_TYPE

	llvm::Type *arg_type(const std::shared_ptr<ast::ASTNode> &type_annotation);

	template<typename... Args> void set_error_state(std::string_view msg, Args &&... args);
};

class LLVMFunction : public ::Function
{
	const llvm::Function &m_function;

  public:
	LLVMFunction(const llvm::Function &f);

	std::string to_string() const override;

	const llvm::Function &impl() const { return m_function; }

	std::vector<uint8_t> serialize() const override { TODO(); }
};
}// namespace codegen