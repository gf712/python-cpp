#include "AST.hpp"
#include "bytecode/BytecodeGenerator.hpp"
#include "bytecode/instructions/Instructions.hpp"
#include "bytecode/instructions/FunctionCall.hpp"
#include "interpreter/Interpreter.hpp"

namespace ast {

#define __AST_NODE_TYPE(x)                                                                     \
	template<> std::shared_ptr<x> as(std::shared_ptr<ASTNode> node)                            \
	{                                                                                          \
		if (node->node_type() == ASTNodeType::x) { return std::static_pointer_cast<x>(node); } \
		return nullptr;                                                                        \
	}
AST_NODE_TYPES
#undef __AST_NODE_TYPE


Register Name::generate(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	auto dst_register = generator.allocate_register();

	if (ctx.has_local_args()) {
		const auto &local_args = ctx.local_args();
		const auto &arg_names = local_args->argument_names();
		if (auto it = std::find(arg_names.begin(), arg_names.end(), m_id[0]);
			it != arg_names.end()) {
			const size_t arg_index = std::distance(arg_names.begin(), it);
			generator.emit<LoadFast>(function_id, dst_register, arg_index, m_id[0]);
			return dst_register;
		}
	}
	generator.emit<LoadName>(function_id, dst_register, m_id[0]);
	return dst_register;
}


Register Constant::generate(size_t function_id, BytecodeGenerator &generator, ASTContext &) const
{
	auto dst_register = generator.allocate_register();
	generator.emit<LoadConst>(function_id, dst_register, m_value);
	return dst_register;
}

Register
	BinaryExpr::generate(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	auto lhs_register = m_lhs->generate(function_id, generator, ctx);
	auto rhs_register = m_rhs->generate(function_id, generator, ctx);
	auto dst_register = generator.allocate_register();

	switch (m_op_type) {
	case OpType::PLUS: {
		generator.emit<Add>(function_id, dst_register, lhs_register, rhs_register);
		return dst_register;
	}
	case OpType::MINUS: {
		generator.emit<Subtract>(function_id, dst_register, lhs_register, rhs_register);
		return dst_register;
	}
	case OpType::MULTIPLY: {
		generator.emit<Multiply>(function_id, dst_register, lhs_register, rhs_register);
		return dst_register;
	}
	case OpType::EXP: {
		generator.emit<Exp>(function_id, dst_register, lhs_register, rhs_register);
		return dst_register;
	}
	case OpType::SLASH:
		TODO()
	case OpType::LEFTSHIFT: {
		generator.emit<LeftShift>(function_id, dst_register, lhs_register, rhs_register);
		return dst_register;
	}
	case OpType::RIGHTSHIFT:
		TODO()
	}
	ASSERT_NOT_REACHED()
}

Register FunctionDefinition::generate(size_t function_id,
	BytecodeGenerator &generator,
	ASTContext &ctx) const
{
	ctx.push_local_args(m_args);

	auto this_function_id = generator.allocate_function();
	m_args->generate(this_function_id, generator, ctx);

	for (const auto &node : m_body) { node->generate(this_function_id, generator, ctx); }

	std::vector<std::string> arg_names;
	for (const auto &arg_name : m_args->argument_names()) { arg_names.push_back(arg_name); }

	generator.emit<MakeFunction>(function_id, m_function_name, this_function_id, arg_names);

	ctx.pop_local_args();
	return {};
}

Register
	Arguments::generate(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	for (const auto &arg : m_args) { arg->generate(function_id, generator, ctx); }
	return {};
}

std::vector<std::string> Arguments::argument_names() const
{
	std::vector<std::string> arg_names;
	for (const auto &arg : m_args) { arg_names.push_back(arg->name()); }
	return arg_names;
}

Register Argument::generate(size_t, BytecodeGenerator &, ASTContext &) const { return {}; }

Register Return::generate(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	auto src_register = m_value->generate(function_id, generator, ctx);
	generator.emit<ReturnValue>(function_id, src_register);
	// return register
	return 0;
}

Register Assign::generate(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	const auto src_register = m_value->generate(function_id, generator, ctx);
	ASSERT(m_targets.size() > 0)
	if (ctx.has_local_args()) {
		const auto &local_args = ctx.local_args();
		const auto &arg_names = local_args->argument_names();
		for (const auto &target : m_targets) {
			for (const auto &var : target->ids()) {
				if (auto it = std::find(arg_names.begin(), arg_names.end(), var);
					it != arg_names.end()) {
					const size_t arg_index = std::distance(arg_names.begin(), it);
					generator.emit<StoreFast>(function_id, arg_index, var, src_register);
				} else {
					generator.emit<StoreName>(function_id, var, src_register);
				}
			}
		}
	} else {
		for (const auto &target : m_targets) {
			for (const auto &var : target->ids()) {
				generator.emit<StoreName>(function_id, var, src_register);
			}
		}
	}
	return src_register;
}

Register Call::generate(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	std::vector<Register> arg_registers;
	arg_registers.reserve(m_args.size());

	auto function_name_ast = as<Name>(m_function);
	if (!function_name_ast) { TODO(); }
	auto func_register = function_name_ast->generate(function_id, generator, ctx);

	for (const auto &arg : m_args) {
		arg_registers.push_back(arg->generate(function_id, generator, ctx));
	}

	generator.emit<FunctionCall>(function_id, func_register, std::move(arg_registers));

	return {};
}

Register If::generate(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	static size_t if_count = 0;

	ASSERT(!m_body.empty())

	auto orelse_start_label =
		generator.make_label(fmt::format("ORELSE_{}", if_count++), function_id);
	auto end_label = generator.make_label(fmt::format("END_{}", if_count), function_id);

	// if
	const auto test_result_register = m_test->generate(function_id, generator, ctx);
	generator.emit<JumpIfFalse>(function_id, test_result_register, orelse_start_label);
	for (const auto &body_statement : m_body) {
		body_statement->generate(function_id, generator, ctx);
	}
	generator.emit<Jump>(function_id, end_label);

	// else
	generator.bind(orelse_start_label);
	for (const auto &orelse_statement : m_orelse) {
		orelse_statement->generate(function_id, generator, ctx);
	}
	generator.bind(end_label);

	return {};
}

Register Compare::generate(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	const auto lhs_reg = m_lhs->generate(function_id, generator, ctx);
	const auto rhs_reg = m_rhs->generate(function_id, generator, ctx);
	const auto result_reg = generator.allocate_register();

	switch (m_op) {
	case OpType::Eq: {
		generator.emit<Equal>(function_id, result_reg, lhs_reg, rhs_reg);
	} break;
	default: {
		TODO()
	}
	}
	return result_reg;
}


Register List::generate(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	std::vector<Register> element_registers;
	element_registers.reserve(m_elements.size());
	const auto result_reg = generator.allocate_register();

	for (const auto &el : m_elements) {
		element_registers.push_back(el->generate(function_id, generator, ctx));
	}

	generator.emit<BuildList>(function_id, result_reg, element_registers);

	return result_reg;
}

}// namespace ast