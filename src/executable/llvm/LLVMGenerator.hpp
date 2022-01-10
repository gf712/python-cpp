#include "ast/AST.hpp"
#include "ast/optimizers/ConstantFolding.hpp"

#include "executable/Function.hpp"

namespace llvm {
    class Module;
}

class LLVMGenerator: public ast::CodeGenerator {
    struct Context;

    public:
    	static std::shared_ptr<Program> compile(std::shared_ptr<ast::ASTNode> node,
		std::vector<std::string> argv,
		compiler::OptimizationLevel lvl);

private:
#define __AST_NODE_TYPE(NodeType) void visit(const ast::NodeType *node) override;
	AST_NODE_TYPES
#undef __AST_NODE_TYPE
};

class LLVMFunction : public ::Function
{
    std::unique_ptr<llvm::Module> m_module;

  public:
	LLVMFunction(std::unique_ptr<llvm::Module>&& module);

	std::string to_string() const override;
};
