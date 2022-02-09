#include "LLVMGenerator.hpp"
#include "LLVMProgram.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace codegen;

namespace {
static AllocaInst *create_entry_block_alloca(llvm::Function *f, const std::string &name, Type *type)
{
	IRBuilder<> b(&f->getEntryBlock(), f->getEntryBlock().begin());
	return b.CreateAlloca(type, 0, name);
}
}// namespace

class LLVMGenerator::LLVMValue : public ast::Value
{
	llvm::Value *m_value;

  public:
	LLVMValue(llvm::Value *value) : ast::Value(value->getName()), m_value(value) {}

	llvm::Value *value() const { return m_value; }
};

struct LLVMGenerator::Context
{
	struct State
	{
		enum class Status { OK, ERROR };
		Status status{ Status::OK };
	};

	struct Scope
	{
		std::map<std::string, llvm::AllocaInst *> lookup_table;
		Scope *parent{ nullptr };
	};

	State state;
	std::stack<Scope> scope_stack;
	std::unique_ptr<LLVMContext> ctx;
	std::unique_ptr<IRBuilder<>> builder;
	std::unique_ptr<Module> module;

	Context(const std::string &module_name)
	{
		ctx = std::make_unique<LLVMContext>();
		builder = std::make_unique<IRBuilder<>>(*ctx);
		module = std::make_unique<Module>(module_name, *ctx);
	}

	// void add_global(StringRef name, llvm::Value* value) {
	// 	module->getOrInsertGlobal(name, value->)
	// }

	void push_stack() { scope_stack.push(Scope{ .parent = &scope_stack.top() }); }
	void pop_stack() { scope_stack.pop(); }

	void add_local(StringRef name, llvm::AllocaInst *value)
	{
		ASSERT(!scope_stack.empty())
		scope_stack.top().lookup_table[name] = value;
	}

	llvm::AllocaInst *get_variable(StringRef name)
	{
		ASSERT(!scope_stack.empty())
		// search in local scope
		if (auto it = scope_stack.top().lookup_table.find(name);
			it != scope_stack.top().lookup_table.end()) {
			return it->second;
		}

		return nullptr;
	}
};

LLVMGenerator::LLVMGenerator() : m_ctx(std::make_unique<Context>("test")) {}

std::shared_ptr<Program> LLVMGenerator::compile(std::shared_ptr<ast::ASTNode> node,
	std::vector<std::string> argv,
	compiler::OptimizationLevel lvl)
{
	auto module = ast::as<ast::Module>(node);
	ASSERT(module)

	auto generator = LLVMGenerator();

	node->codegen(&generator);

	// if (generator.m_ctx->state.status == LLVMGenerator::Context::State::Status::ERROR) {
	// 	return nullptr;
	// }

	return std::make_shared<LLVMProgram>(std::move(generator.m_ctx->module),
		std::move(generator.m_ctx->ctx),
		module->filename(),
		argv);
}

LLVMGenerator::LLVMValue *LLVMGenerator::generate(const ast::ASTNode *node)
{
	if (m_ctx->state.status == LLVMGenerator::Context::State::Status::ERROR) { return nullptr; }
	auto *value = node->codegen(this);
	if (!value) return nullptr;
	ASSERT(value);
	return static_cast<LLVMGenerator::LLVMValue *>(value);
}

ast::Value *LLVMGenerator::create_value(llvm::Value *value)
{
	return m_values.emplace_back(std::make_unique<LLVMGenerator::LLVMValue>(value)).get();
}

ast::Value *LLVMGenerator::visit(const ast::Argument *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Arguments *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Attribute *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Assign *node)
{
	auto *value_to_store = generate(node->value().get());
	if (!value_to_store) { return nullptr; }
	for (const auto &target : node->targets()) {
		if (auto ast_name = ast::as<ast::Name>(target)) {
			ASSERT(ast_name->ids().size() == 1)
			const auto &var_name = ast_name->ids()[0];
			llvm::Function *f = m_ctx->builder->GetInsertBlock()->getParent();
			Type *type = value_to_store->value()->getType();
			if (auto it = m_ctx->scope_stack.top().lookup_table.find(var_name);
				it != m_ctx->scope_stack.top().lookup_table.end()) {
				if (type != it->second->getType()) {
					std::string repr1;
					raw_string_ostream out1{ repr1 };
					type->print(out1);

					std::string repr2;
					raw_string_ostream out2{ repr2 };
					type->print(out2);

					set_error_state("Cannot assign value of type '{}' to variable of type '{}",
						out1.str(),
						out2.str());
				}
			}
			auto *alloca_inst = create_entry_block_alloca(f, var_name, type);
			m_ctx->add_local(var_name, alloca_inst);
			return create_value(m_ctx->builder->CreateStore(value_to_store->value(), alloca_inst));
		} else {
			set_error_state("Can only assign to ast::Name");
			return nullptr;
		}
	}

	TODO();
	return nullptr;
}

ast::Value *LLVMGenerator::visit(const ast::Assert *node)
{
	set_error_state("ast::Assert node not implemented");
	return nullptr;
}

ast::Value *LLVMGenerator::visit(const ast::AugAssign *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::BinaryExpr *node)
{
	switch (node->op_type()) {
	case ast::BinaryOpType::PLUS: {
		auto *lhs = generate(node->lhs().get());
		if (!lhs) return nullptr;
		auto *rhs = generate(node->rhs().get());
		if (!rhs) return nullptr;
		return create_value(m_ctx->builder->CreateAdd(lhs->value(), rhs->value()));
	} break;
	case ast::BinaryOpType::MINUS: {
		TODO();
	} break;
	case ast::BinaryOpType::MODULO: {
		TODO();
	} break;
	case ast::BinaryOpType::MULTIPLY: {
		TODO();
	} break;
	case ast::BinaryOpType::EXP: {
		TODO();
	} break;
	case ast::BinaryOpType::SLASH: {
		TODO();
	} break;
	case ast::BinaryOpType::FLOORDIV: {
		TODO();
	} break;
	case ast::BinaryOpType::LEFTSHIFT: {
		TODO();
	} break;
	case ast::BinaryOpType::RIGHTSHIFT: {
		TODO();
	} break;
	}
}

ast::Value *LLVMGenerator::visit(const ast::BoolOp *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Call *node)
{
	if (node->function()->node_type() != ast::ASTNodeType::Name) {
		set_error_state("arg type is not a Name AST node cannot call function");
		return nullptr;
	}
	auto function_name_node = std::static_pointer_cast<ast::Name>(node->function());
	ASSERT(function_name_node->ids().size() == 1)

	const auto &function_name = function_name_node->ids()[0];

	auto *f = [this, &function_name]() -> llvm::Function * {
		for (auto &f : m_ctx->module->functions()) {
			if (function_name == f.getName()) { return &f; }
		}
		return nullptr;
	}();

	if (!f) {
		set_error_state("did not find function {}", function_name);
		return nullptr;
	}

	TODO();
	return nullptr;
	// m_ctx->last_value = m_ctx->builder->CreateCall(f->getFunctionType(), f);
}

ast::Value *LLVMGenerator::visit(const ast::ClassDefinition *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Compare *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Constant *node) { return nullptr; }

ast::Value *LLVMGenerator::visit(const ast::Dict *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::ExceptHandler *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Delete *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Global *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::IfExpr *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::NamedExpr *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Starred *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::With *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::WithItem *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::For *node) { return nullptr; }

llvm::Type *LLVMGenerator::arg_type(const std::shared_ptr<ast::ASTNode> &type_annotation)
{
	if (auto name = ast::as<ast::Name>(type_annotation)) {
		if (name->ids().size() != 1) {
			set_error_state("arg type is not a AST node with one id, cannot determine arg type");
			return nullptr;
		}
		const auto &type_name = name->ids()[0];
		if (type_name == "int") {
			return Type::getInt64Ty(*m_ctx->ctx);
		} else if (type_name == "float") {
			return Type::getDoubleTy(*m_ctx->ctx);
		} else {
			set_error_state("arg type {} currenlty not supported", type_name);
			return nullptr;
		}
	} else {
		set_error_state("arg type is not a Name AST node, cannot determine arg type");
	}
	return nullptr;
}// namespace

ast::Value *LLVMGenerator::visit(const ast::FunctionDefinition *node)
{
	std::vector<Type *> arg_types;

	for (const auto &arg : node->args()->args()) {
		const auto &type_annotation = arg->annotation();
		if (!type_annotation) {
			set_error_state("arg is not type annotated, cannot determine arg type");
			return nullptr;
		}
		if (auto *type = arg_type(type_annotation)) {
			arg_types.push_back(type);
		} else {
			return nullptr;
		}
	}

	ASSERT(arg_types.size() == node->args()->args().size())

	if (!node->returns()) {
		// TODO: this would require type checking through the whole function
		set_error_state("empty return type annotation not implemented");
		return nullptr;
	}

	const auto &return_type_annotation = node->returns();
	Type *return_type = nullptr;
	if (auto *type = arg_type(return_type_annotation)) {
		return_type = type;
	} else {
		return nullptr;
	}

	auto *FT = FunctionType::get(return_type, arg_types, false);
	auto *F = llvm::Function::Create(
		FT, llvm::Function::ExternalLinkage, node->name(), m_ctx->module.get());
	BasicBlock *BB = BasicBlock::Create(*m_ctx->ctx, "entry", F);
	m_ctx->builder->SetInsertPoint(BB);

	m_ctx->push_stack();

	{
		size_t i = 0;
		for (const auto &arg : node->args()->args()) {
			auto *llvm_arg = F->getArg(i);
			llvm_arg->setName(arg->name());
			auto *allocation = create_entry_block_alloca(F, arg->name(), arg_types[i]);
			m_ctx->add_local(arg->name(), allocation);
			m_ctx->builder->CreateStore(llvm_arg, allocation);
			i++;
		}
	}

	for (const auto &statement : node->body()) { generate(statement.get()); }

	// m_ctx->pop_stack();
	// {
	// 	std::string repr;
	// 	raw_string_ostream out{ repr };
	// 	F->print(out);
	// 	spdlog::debug("Compiled LLVM function:\n{}", out.str());
	// }

	// FIXME: For some reason this reports an error but does not return a error message
	// {
	// 	std::string repr;
	// 	raw_string_ostream out{ repr };
	// 	auto success = verifyFunction(*F, &out);
	// 	if (!success) { spdlog::error("Failed to compile to LLVM IR: {}", out.str()); }
	// 	ASSERT(success)
	// }

	return create_value(F);
}

ast::Value *LLVMGenerator::visit(const ast::If *node)
{
	auto *condition = generate(node->test().get());

	auto *f = m_ctx->builder->GetInsertBlock()->getParent();

	auto *ThenBB = BasicBlock::Create(*m_ctx->ctx, "then", f);
	auto *ElseBB = BasicBlock::Create(*m_ctx->ctx, "else");
	auto *MergeBB = BasicBlock::Create(*m_ctx->ctx, "ifcont");

	m_ctx->builder->CreateCondBr(condition->value(), ThenBB, ElseBB);

	// then
	m_ctx->builder->SetInsertPoint(ThenBB);
	for (const auto &statement : node->body()) { generate(statement.get()); }
	m_ctx->builder->CreateBr(MergeBB);

	// else
	f->getBasicBlockList().push_back(ElseBB);
	m_ctx->builder->SetInsertPoint(ElseBB);
	for (const auto &statement : node->orelse()) { generate(statement.get()); }

	m_ctx->builder->CreateBr(MergeBB);
	ElseBB = m_ctx->builder->GetInsertBlock();

	// merge block
	f->getBasicBlockList().push_back(MergeBB);
	m_ctx->builder->SetInsertPoint(MergeBB);
	// auto phi_node = m_ctx->builder->CreatePHI();
	TODO();

	return nullptr;
}

ast::Value *LLVMGenerator::visit(const ast::Import *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Keyword *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::List *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Module *node)
{
	m_ctx->module->setSourceFileName(node->filename());
	m_ctx->module->setModuleIdentifier("test");

	for (const auto &statement : node->body()) {
		statement->codegen(this);
		// if (m_ctx->state.status == LLVMGenerator::Context::State::Status::ERROR) { return
		// nullptr; }
	}

	std::string repr;
	raw_string_ostream out{ repr };
	m_ctx->module->print(out, nullptr);
	spdlog::debug("LLVM module:\n{}", out.str());

	return nullptr;
}

ast::Value *LLVMGenerator::visit(const ast::Name *node)
{
	const auto &var_name = node->ids()[0];
	ASSERT(m_ctx->builder->GetInsertBlock())
	auto *current_func = m_ctx->builder->GetInsertBlock()->getParent();
	ASSERT(current_func)
	auto *var_alloca = m_ctx->get_variable(var_name);

	if (!var_alloca) {
		set_error_state("Could not find '{}' in function", var_name);
		return nullptr;
	}

	return create_value(
		m_ctx->builder->CreateLoad(var_alloca->getAllocatedType(), var_alloca, var_name));
}

ast::Value *LLVMGenerator::visit(const ast::Pass *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Raise *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Return *node)
{
	auto *return_value = generate(node->value().get());
	m_ctx->builder->CreateRet(return_value->value());
	return return_value;
}

ast::Value *LLVMGenerator::visit(const ast::Subscript *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Try *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::Tuple *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::UnaryExpr *node) { TODO(); }

ast::Value *LLVMGenerator::visit(const ast::While *node) { TODO(); }

template<typename... Args>
void LLVMGenerator::set_error_state(std::string_view msg, Args &&... args)
{
	spdlog::debug(msg, std::forward<Args>(args)...);
	m_ctx->state.status = Context::State::Status::ERROR;
}

LLVMFunction::LLVMFunction(const llvm::Function &f)
	: Function(0, 0, f.getName().str(), FunctionExecutionBackend::LLVM), m_function(f)
{}

std::string LLVMFunction::to_string() const
{
	std::string repr;
	raw_string_ostream out{ repr };
	m_function.print(out, nullptr);
	return out.str();
}