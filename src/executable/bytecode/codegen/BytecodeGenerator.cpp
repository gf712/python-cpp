#include "BytecodeGenerator.hpp"

#include "executable/bytecode/instructions/ClearExceptionState.hpp"
#include "executable/bytecode/instructions/FunctionCall.hpp"
#include "executable/bytecode/instructions/FunctionCallWithKeywords.hpp"
#include "executable/bytecode/instructions/ImportName.hpp"
#include "executable/bytecode/instructions/InplaceAdd.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"
#include "executable/bytecode/instructions/IsOp.hpp"
#include "executable/bytecode/instructions/JumpForward.hpp"
#include "executable/bytecode/instructions/JumpIfFalseOrPop.hpp"
#include "executable/bytecode/instructions/JumpIfNotExceptionMatch.hpp"
#include "executable/bytecode/instructions/JumpIfTrue.hpp"
#include "executable/bytecode/instructions/JumpIfTrueOrPop.hpp"
#include "executable/bytecode/instructions/LoadAssertionError.hpp"
#include "executable/bytecode/instructions/LoadAttr.hpp"
#include "executable/bytecode/instructions/LoadBuildClass.hpp"
#include "executable/bytecode/instructions/LoadMethod.hpp"
#include "executable/bytecode/instructions/LoadName.hpp"
#include "executable/bytecode/instructions/MethodCall.hpp"
#include "executable/bytecode/instructions/NotEqual.hpp"
#include "executable/bytecode/instructions/RaiseVarargs.hpp"
#include "executable/bytecode/instructions/ReturnValue.hpp"
#include "executable/bytecode/instructions/SetupExceptionHandling.hpp"
#include "executable/bytecode/instructions/StoreAttr.hpp"
#include "executable/bytecode/instructions/StoreName.hpp"
#include "executable/bytecode/instructions/Unary.hpp"
#include "executable/bytecode/instructions/UnpackSequence.hpp"

#include "ast/optimizers/ConstantFolding.hpp"
#include "executable/FunctionBlock.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"

using namespace ast;

namespace codegen {

void BytecodeGenerator::visit(const Name *node)
{
	m_last_register = allocate_register();

	if (m_ctx.has_local_args()) {
		const auto &local_args = m_ctx.local_args();
		const auto &arg_names = local_args->argument_names();
		if (auto it = std::find(arg_names.begin(), arg_names.end(), node->ids()[0]);
			it != arg_names.end()) {
			const size_t arg_index = std::distance(arg_names.begin(), it);
			emit<LoadFast>(m_last_register, arg_index, node->ids()[0]);
			return;
		}
	}
	emit<LoadName>(m_last_register, node->ids()[0]);
}


void BytecodeGenerator::visit(const Constant *node)
{
	auto dst_register = allocate_register();
	emit<LoadConst>(dst_register, node->value());
	m_last_register = dst_register;
}

void BytecodeGenerator::visit(const BinaryExpr *node)
{
	auto lhs_register = generate(node->lhs().get(), m_function_id);
	auto rhs_register = generate(node->rhs().get(), m_function_id);
	auto dst_register = allocate_register();

	switch (node->op_type()) {
	case BinaryOpType::PLUS: {
		emit<Add>(dst_register, lhs_register, rhs_register);
	} break;
	case BinaryOpType::MINUS: {
		emit<Subtract>(dst_register, lhs_register, rhs_register);
	} break;
	case BinaryOpType::MULTIPLY: {
		emit<Multiply>(dst_register, lhs_register, rhs_register);
	} break;
	case BinaryOpType::EXP: {
		emit<Exp>(dst_register, lhs_register, rhs_register);
	} break;
	case BinaryOpType::MODULO: {
		emit<Modulo>(dst_register, lhs_register, rhs_register);
	} break;
	case BinaryOpType::SLASH:
		TODO();
	case BinaryOpType::FLOORDIV:
		TODO();
	case BinaryOpType::LEFTSHIFT: {
		emit<LeftShift>(dst_register, lhs_register, rhs_register);
	} break;
	case BinaryOpType::RIGHTSHIFT:
		TODO();
	}

	m_last_register = dst_register;
}

void BytecodeGenerator::visit(const FunctionDefinition *node)
{
	if (!node->decorator_list().empty()) { TODO(); }
	m_ctx.push_local_args(node->args());
	auto this_function_info = allocate_function();

	auto *block = allocate_block(this_function_info.function_id);
	auto *old_block = m_current_block;
	set_insert_point(block);

	generate(node->args().get(), this_function_info.function_id);

	for (const auto &node : node->body()) { generate(node.get(), this_function_info.function_id); }

	// always return None
	// this can be optimised away later on
	auto none_value_register = allocate_register();
	emit<LoadConst>(none_value_register, NameConstant{ NoneType{} });
	emit<ReturnValue>(none_value_register);

	std::vector<std::string> arg_names;
	for (const auto &arg_name : node->args()->argument_names()) { arg_names.push_back(arg_name); }

	set_insert_point(old_block);

	emit<MakeFunction>(this_function_info.function_id, node->name(), arg_names);

	m_ctx.pop_local_args();
	m_last_register = Register{};
}

void BytecodeGenerator::visit(const Arguments *node)
{
	if (!node->kwonlyargs().empty()) { TODO(); }
	if (node->vararg()) { TODO(); }
	if (node->kwarg()) { TODO(); }

	for (const auto &arg : node->args()) { generate(arg.get(), m_function_id); }

	m_last_register = Register{};
}

void BytecodeGenerator::visit(const Argument *) { m_last_register = Register{}; }

void BytecodeGenerator::visit(const Starred *) { TODO(); }

void BytecodeGenerator::visit(const Return *node)
{
	auto src_register = generate(node->value().get(), m_function_id);
	emit<ReturnValue>(src_register);
	m_last_register = Register{};
}

void BytecodeGenerator::visit(const Assign *node)
{
	const auto src_register = generate(node->value().get(), m_function_id);
	ASSERT(node->targets().size() > 0)

	for (const auto &target : node->targets()) {
		if (auto ast_name = as<Name>(target)) {
			for (const auto &var : ast_name->ids()) {
				if (m_ctx.has_local_args()) {
					const auto &local_args = m_ctx.local_args();
					const auto &arg_names = local_args->argument_names();
					if (auto it = std::find(arg_names.begin(), arg_names.end(), var);
						it != arg_names.end()) {
						const size_t arg_index = std::distance(arg_names.begin(), it);
						emit<StoreFast>(arg_index, var, src_register);
						continue;
					}
				}
				emit<StoreName>(var, src_register);
			}
		} else if (auto ast_attr = as<Attribute>(target)) {
			auto dst_register = generate(ast_attr->value().get(), m_function_id);
			emit<StoreAttr>(dst_register, src_register, ast_attr->attr());
		} else if (auto ast_tuple = as<Tuple>(target)) {
			std::vector<Register> dst_registers;
			for (size_t i = 0; i < ast_tuple->elements().size(); ++i) {
				dst_registers.push_back(allocate_register());
			}
			emit<UnpackSequence>(dst_registers, src_register);
			size_t idx{ 0 };
			for (const auto &dst_register : dst_registers) {
				const auto &el = ast_tuple->elements()[idx++];
				emit<StoreName>(as<Name>(el)->ids()[0], dst_register);
			}
		} else {
			TODO();
		}
	}
	m_last_register = src_register;
}

void BytecodeGenerator::visit(const Call *node)
{
	std::vector<Register> arg_registers;
	std::vector<Register> keyword_registers;
	std::vector<std::string> keywords;

	arg_registers.reserve(node->args().size());
	keyword_registers.reserve(node->keywords().size());
	keywords.reserve(node->keywords().size());

	auto func_register = generate(node->function().get(), m_function_id);

	for (const auto &arg : node->args()) {
		arg_registers.push_back(generate(arg.get(), m_function_id));
	}

	for (const auto &keyword : node->keywords()) {
		keyword_registers.push_back(generate(keyword.get(), m_function_id));
		auto keyword_argname = keyword->arg();
		if (!keyword_argname.has_value()) { TODO(); }
		keywords.push_back(*keyword_argname);
	}

	if (node->function()->node_type() == ASTNodeType::Attribute) {
		auto attr_name = as<Attribute>(node->function())->attr();
		emit<MethodCall>(func_register, attr_name, std::move(arg_registers));
	} else {
		if (keyword_registers.empty()) {
			emit<FunctionCall>(func_register, std::move(arg_registers));
		} else {
			emit<FunctionCallWithKeywords>(func_register,
				std::move(arg_registers),
				std::move(keyword_registers),
				std::move(keywords));
		}
	}

	m_last_register = Register{};
}

void BytecodeGenerator::visit(const If *node)
{
	static size_t if_count = 0;

	ASSERT(!node->body().empty())

	auto orelse_start_label = make_label(fmt::format("ORELSE_{}", if_count), m_function_id);
	auto end_label = make_label(fmt::format("END_{}", if_count++), m_function_id);

	// if
	const auto test_result_register = generate(node->test().get(), m_function_id);
	emit<JumpIfFalse>(test_result_register, orelse_start_label);
	for (const auto &body_statement : node->body()) {
		generate(body_statement.get(), m_function_id);
	}
	emit<Jump>(end_label);

	// else
	bind(*orelse_start_label);
	for (const auto &orelse_statement : node->orelse()) {
		generate(orelse_statement.get(), m_function_id);
	}
	bind(*end_label);

	m_last_register = Register{};
}

void BytecodeGenerator::visit(const For *node)
{
	static size_t for_loop_count = 0;

	ASSERT(!node->body().empty())

	auto forloop_start_label =
		make_label(fmt::format("FOR_START_{}", for_loop_count), m_function_id);
	auto forloop_end_label = make_label(fmt::format("FOR_END_{}", for_loop_count++), m_function_id);

	// generate the iterator
	const auto iterator_func_register = generate(node->iter().get(), m_function_id);

	auto iterator_register = allocate_register();
	auto iter_variable_register = allocate_register();

	// call the __iter__ implementation
	emit<GetIter>(iterator_register, iterator_func_register);

	// call the __next__ implementation
	auto target_ids = as<Name>(node->target())->ids();
	if (target_ids.size() != 1) { TODO(); }
	auto target_name = target_ids[0];

	bind(*forloop_start_label);
	emit<ForIter>(iter_variable_register, iterator_register, target_name, forloop_end_label);

	generate(node->target().get(), m_function_id);

	// body
	for (const auto &el : node->body()) { generate(el.get(), m_function_id); }
	emit<Jump>(forloop_start_label);

	// orelse
	bind(*forloop_end_label);
	for (const auto &el : node->orelse()) { generate(el.get(), m_function_id); }

	m_last_register = Register{};
}

void BytecodeGenerator::visit(const While *node)
{
	static size_t while_loop_count = 0;

	ASSERT(!node->body().empty())

	auto while_loop_start_label =
		make_label(fmt::format("WHILE_START_{}", while_loop_count), m_function_id);
	auto while_loop_end_label =
		make_label(fmt::format("WHILE_END_{}", while_loop_count++), m_function_id);

	// test
	bind(*while_loop_start_label);
	const auto test_result_register = generate(node->test().get(), m_function_id);
	emit<JumpIfFalse>(test_result_register, while_loop_end_label);

	// body
	for (const auto &el : node->body()) { generate(el.get(), m_function_id); }
	emit<Jump>(while_loop_start_label);

	// orelse
	bind(*while_loop_end_label);
	for (const auto &el : node->orelse()) { generate(el.get(), m_function_id); }

	m_last_register = Register{};
}

void BytecodeGenerator::visit(const Compare *node)
{
	const auto lhs_reg = generate(node->lhs().get(), m_function_id);
	const auto rhs_reg = generate(node->rhs().get(), m_function_id);
	const auto result_reg = allocate_register();

	switch (node->op()) {
	case Compare::OpType::Eq: {
		emit<Equal>(result_reg, lhs_reg, rhs_reg);
	} break;
	case Compare::OpType::NotEq: {
		emit<NotEqual>(result_reg, lhs_reg, rhs_reg);
	} break;
	case Compare::OpType::Lt: {
		emit<LessThan>(result_reg, lhs_reg, rhs_reg);
	} break;
	case Compare::OpType::LtE: {
		emit<LessThanEquals>(result_reg, lhs_reg, rhs_reg);
	} break;
	case Compare::OpType::Gt: {
		TODO();
	} break;
	case Compare::OpType::GtE: {
		TODO();
	} break;
	case Compare::OpType::Is: {
		emit<IsOp>(result_reg, lhs_reg, rhs_reg, false);
	} break;
	case Compare::OpType::IsNot: {
		emit<IsOp>(result_reg, lhs_reg, rhs_reg, true);
	} break;
	case Compare::OpType::In: {
		TODO();
	} break;
	case Compare::OpType::NotIn: {
		TODO();
	} break;
	}
	m_last_register = result_reg;
}

void BytecodeGenerator::visit(const List *node)
{
	std::vector<Register> element_registers;
	element_registers.reserve(node->elements().size());

	for (const auto &el : node->elements()) {
		element_registers.push_back(generate(el.get(), m_function_id));
	}

	const auto result_reg = allocate_register();
	emit<BuildList>(result_reg, element_registers);

	m_last_register = result_reg;
}

void BytecodeGenerator::visit(const Tuple *node)
{
	std::vector<Register> element_registers;
	element_registers.reserve(node->elements().size());

	for (const auto &el : node->elements()) {
		element_registers.push_back(generate(el.get(), m_function_id));
	}

	const auto result_reg = allocate_register();
	emit<BuildTuple>(result_reg, element_registers);

	m_last_register = result_reg;
}

void BytecodeGenerator::visit(const ClassDefinition *node)
{
	if (!node->decorator_list().empty()) { TODO(); }

	size_t class_id;
	{
		auto this_class_info = allocate_function();
		class_id = this_class_info.function_id;

		auto *block = allocate_block(class_id);
		auto *old_block = m_current_block;
		set_insert_point(block);

		const auto name_register = allocate_register();
		const auto qualname_register = allocate_register();
		const auto return_none_register = allocate_register();

		// class definition preamble, a la CPython
		emit<LoadName>(name_register, "__name__");
		emit<StoreName>("__module__", name_register);
		emit<LoadConst>(qualname_register, String{ "A" });
		emit<StoreName>("__qualname__", qualname_register);

		// the actual class definition
		for (const auto &el : node->body()) { generate(el.get(), class_id); }

		emit<LoadConst>(return_none_register, NameConstant{ NoneType{} });
		emit<ReturnValue>(return_none_register);

		set_insert_point(old_block);
	}

	std::vector<Register> arg_registers;
	arg_registers.reserve(2 + node->bases().size());
	std::vector<Register> kwarg_registers;
	std::vector<std::string> keyword_names;

	const auto builtin_build_class_register = allocate_register();
	const auto class_location_register = allocate_register();
	const auto class_name_register = allocate_register();

	arg_registers.push_back(class_location_register);
	arg_registers.push_back(class_name_register);

	for (const auto &base : node->bases()) {
		arg_registers.push_back(generate(base.get(), m_function_id));
	}
	for (const auto &keyword : node->keywords()) {
		kwarg_registers.push_back(generate(keyword.get(), m_function_id));
		auto keyword_argname = keyword->arg();
		if (!keyword_argname.has_value()) { TODO(); }
		keyword_names.push_back(*keyword_argname);
	}

	emit<LoadBuildClass>(builtin_build_class_register);
	emit<LoadConst>(class_name_register, String{ node->name() });
	emit<LoadConst>(class_location_register, Number{ static_cast<int64_t>(class_id) });

	if (kwarg_registers.empty()) {
		emit<FunctionCall>(builtin_build_class_register, std::move(arg_registers));
	} else {
		emit<FunctionCallWithKeywords>(builtin_build_class_register,
			std::move(arg_registers),
			std::move(kwarg_registers),
			std::move(keyword_names));
	}

	emit<StoreName>(node->name(), Register{ 0 });

	m_last_register = Register{};
}

void BytecodeGenerator::visit(const Dict *node)
{
	ASSERT(node->keys().size() == node->values().size())

	std::vector<Register> key_registers;
	std::vector<Register> value_registers;

	for (const auto &key : node->keys()) {
		key_registers.push_back(generate(key.get(), m_function_id));
	}
	for (const auto &value : node->values()) {
		value_registers.push_back(generate(value.get(), m_function_id));
	}

	const auto result_reg = allocate_register();
	emit<BuildDict>(result_reg, key_registers, value_registers);

	m_last_register = result_reg;
}

void BytecodeGenerator::visit(const Attribute *node)
{
	auto this_value_register = generate(node->value().get(), m_function_id);

	const auto *parent_node = m_ctx.parent_nodes()[m_ctx.parent_nodes().size() - 2];
	auto parent_node_type = parent_node->node_type();

	// the parent is a Call AST node and this is the function that is being called, then this
	// must be a method "foo.bar()" -> .bar() is the function being called by parent AST node
	// and this attribute
	if (parent_node_type == ASTNodeType::Call
		&& static_cast<const Call *>(parent_node)->function().get() == node) {
		auto method_name_register = allocate_register();
		emit<LoadMethod>(method_name_register, this_value_register, node->attr());
		m_last_register = method_name_register;
	} else if (node->context() == ContextType::LOAD) {
		auto attribute_value_register = allocate_register();
		emit<LoadAttr>(attribute_value_register, this_value_register, node->attr());
		m_last_register = attribute_value_register;
	} else if (node->context() == ContextType::STORE) {
		auto attribute_value_register = allocate_register();
		emit<LoadAttr>(attribute_value_register, this_value_register, node->attr());
		m_last_register = attribute_value_register;
	} else {
		TODO();
	}
}

void BytecodeGenerator::visit(const Keyword *node)
{
	m_last_register = generate(node->value().get(), m_function_id);
}

void BytecodeGenerator::visit(const AugAssign *node)
{
	const auto lhs_register = generate(node->target().get(), m_function_id);
	const auto rhs_register = generate(node->value().get(), m_function_id);
	switch (node->op()) {
	case BinaryOpType::PLUS: {
		emit<InplaceAdd>(lhs_register, rhs_register);
	} break;
	case BinaryOpType::MINUS: {
		TODO();
	} break;
	case BinaryOpType::MULTIPLY: {
		TODO();
	} break;
	case BinaryOpType::EXP: {
		TODO();
	} break;
	case BinaryOpType::MODULO: {
		TODO();
	} break;
	case BinaryOpType::SLASH:
		TODO();
	case BinaryOpType::FLOORDIV:
		TODO();
	case BinaryOpType::LEFTSHIFT: {
		TODO();
	} break;
	case BinaryOpType::RIGHTSHIFT:
		TODO();
	}

	if (auto named_target = as<Name>(node->target())) {
		if (named_target->ids().size() != 1) { TODO(); }
		emit<StoreName>(named_target->ids()[0], lhs_register);
	} else if (auto attr = as<Attribute>(node->target())) {
		auto obj_reg = generate(attr->value().get(), m_function_id);
		emit<StoreAttr>(obj_reg, lhs_register, attr->attr());
	} else {
		TODO();
	}
	m_last_register = lhs_register;
}

void BytecodeGenerator::visit(const Import *node)
{
	auto module_register = allocate_register();
	emit<ImportName>(module_register, node->names());
	if (node->asname().has_value()) {
		emit<StoreName>(*node->asname(), module_register);
	} else {
		emit<StoreName>(node->names()[0], module_register);
	}
	m_last_register = Register{};
}

void BytecodeGenerator::visit(const Module *node)
{
	Register last{ 0 };
	for (const auto &statement : node->body()) { last = generate(statement.get(), m_function_id); }

	emit<ReturnValue>(last);
	m_last_register = Register{ 0 };
}

void BytecodeGenerator::visit(const Subscript *) { TODO(); }

void BytecodeGenerator::visit(const Raise *) { TODO(); }

void BytecodeGenerator::visit(const With *) { TODO(); }

void BytecodeGenerator::visit(const WithItem *node)
{
	const auto ctx_expr = generate(node->context_expr().get(), m_function_id);

	if (node->optional_vars()) {
		ASSERT(as<Name>(node->context_expr()))
		ASSERT(as<Name>(node->context_expr())->ids().size() == 1)
		emit<StoreName>(as<Name>(node->context_expr())->ids()[0], ctx_expr);
	}
}

void BytecodeGenerator::visit(const IfExpr *) { TODO(); }

namespace {
	size_t list_node_distance(const std::list<InstructionBlock> &block_list,
		InstructionBlock *start,
		InstructionBlock *end)
	{
		if (start == end) { return 0; }
		std::optional<size_t> start_idx;
		std::optional<size_t> end_idx;

		for (size_t idx = 0; const auto &el : block_list) {
			if (&el == start) {
				start_idx = idx;
			} else if (&el == end) {
				end_idx = idx;
			}
			idx++;
		}

		ASSERT(start_idx)
		ASSERT(end_idx)
		return *end_idx - *start_idx;
	}
}// namespace

void BytecodeGenerator::visit(const Try *node)
{
	auto *body_block = allocate_block(m_function_id);

	std::vector<InstructionBlock *> exception_handler_blocks;
	exception_handler_blocks.resize(node->handlers().size() * 2);
	std::generate(exception_handler_blocks.begin(), exception_handler_blocks.end(), [this]() {
		return allocate_block(m_function_id);
	});

	auto *finally_block = allocate_block(m_function_id);

	emit<SetupExceptionHandling>();
	set_insert_point(body_block);

	for (const auto &statement : node->body()) { generate(statement.get(), m_function_id); }

	emit<JumpForward>(list_node_distance(function(m_function_id), body_block, finally_block));

	for (size_t idx = 0; const auto &handler : node->handlers()) {
		auto *exception_handler_block = exception_handler_blocks[idx];
		set_insert_point(exception_handler_block);
		auto exception_type_reg = generate(handler->type().get(), m_function_id);
		emit<JumpIfNotExceptionMatch>(exception_type_reg);
		idx++;
		set_insert_point(exception_handler_blocks[idx]);
		for (const auto &el : handler->body()) { generate(el.get(), m_function_id); }
		emit<ClearExceptionState>();
		emit<JumpForward>(list_node_distance(
			function(m_function_id), exception_handler_blocks[idx], finally_block));
		idx++;
	}

	set_insert_point(finally_block);
	for (const auto &statement : node->finalbody()) { generate(statement.get(), m_function_id); }

	m_last_register = Register{ 0 };
}

void BytecodeGenerator::visit(const ExceptHandler *) { TODO(); }

void BytecodeGenerator::visit(const Global *) { TODO(); }

void BytecodeGenerator::visit(const Delete *) { TODO(); }

void BytecodeGenerator::visit(const UnaryExpr *node)
{
	const auto source_register = generate(node->operand().get(), m_function_id);
	const auto dst_register = allocate_register();
	switch (node->op_type()) {
	case UnaryOpType::ADD: {
		emit<UnaryPositive>(dst_register, source_register);
	} break;
	case UnaryOpType::SUB: {
		emit<UnaryNegative>(dst_register, source_register);
	} break;
	case UnaryOpType::INVERT: {
		emit<UnaryInvert>(dst_register, source_register);
	} break;
	case UnaryOpType::NOT: {
		emit<UnaryNot>(dst_register, source_register);
	} break;
	}

	m_last_register = dst_register;
}

void BytecodeGenerator::visit(const BoolOp *node)
{
	static size_t bool_op_count = 0;
	auto end_label = make_label(fmt::format("BOOL_OP_END_{}", bool_op_count++), m_function_id);
	Register result_register = allocate_register();
	Register last_result;
	switch (node->op()) {
	case BoolOp::OpType::And: {
		auto it = node->values().begin();
		auto end = node->values().end();
		while (std::next(it) != end) {
			last_result = generate((*it).get(), m_function_id);
			emit<JumpIfFalseOrPop>(last_result, result_register, end_label);
			it++;
		}
		last_result = generate((*it).get(), m_function_id);
	} break;
	case BoolOp::OpType::Or: {
		auto it = node->values().begin();
		auto end = node->values().end();
		while (std::next(it) != end) {
			last_result = generate((*it).get(), m_function_id);
			emit<JumpIfTrueOrPop>(last_result, result_register, end_label);
			it++;
		}
		last_result = generate((*it).get(), m_function_id);
	}
	}
	emit<Move>(result_register, last_result);

	bind(*end_label);
	m_last_register = result_register;
}

void BytecodeGenerator::visit(const Assert *node)
{
	static size_t assert_count = 0;
	auto end_label = make_label(fmt::format("END_{}", assert_count++), m_function_id);

	const auto test_result_register = generate(node->test().get(), m_function_id);

	emit<JumpIfTrue>(test_result_register, end_label);

	const auto assertion_function_register = allocate_register();
	emit<LoadAssertionError>(assertion_function_register);

	const auto msg_register = [this, &node]() -> std::optional<Register> {
		if (node->msg()) {
			const auto msg_register = generate(node->msg().get(), m_function_id);
			return msg_register;
		} else {
			return {};
		}
	}();

	if (msg_register.has_value()) {
		emit<RaiseVarargs>(assertion_function_register, *msg_register);
	} else {
		emit<RaiseVarargs>(assertion_function_register);
	}
	bind(*end_label);

	m_last_register = Register{};
}

void BytecodeGenerator::visit(const Pass *) { m_last_register = Register{}; }

void BytecodeGenerator::visit(const NamedExpr *) { TODO(); }

FunctionInfo::FunctionInfo(size_t function_id_, FunctionBlock &f, BytecodeGenerator *generator_)
	: function_id(function_id_), function(f), generator(generator_)
{
	generator->enter_function();
}

FunctionInfo::~FunctionInfo() { generator->exit_function(function_id); }

BytecodeGenerator::BytecodeGenerator() : m_frame_register_count({ start_register })
{
	allocate_function();
	m_current_block = &m_functions.back().blocks.back();
}

BytecodeGenerator::~BytecodeGenerator() {}

void BytecodeGenerator::exit_function(size_t function_id)
{
	ASSERT(function_id < m_functions.size())
	auto function = std::next(m_functions.begin(), function_id);
	function->metadata.register_count = register_count();
	m_frame_register_count.pop_back();
}

FunctionInfo BytecodeGenerator::allocate_function()
{
	auto &new_func = m_functions.emplace_back();
	// allocate the first block
	new_func.blocks.emplace_back();
	new_func.metadata.function_name = std::to_string(m_functions.size() - 1);
	return FunctionInfo{ m_functions.size() - 1, new_func, this };
}

void BytecodeGenerator::relocate_labels(const FunctionBlocks &functions)
{
	for (const auto &function : functions) {
		size_t instruction_idx{ 0 };
		for (const auto &block : function.blocks) {
			for (const auto &ins : block) { ins->relocate(*this, instruction_idx++); }
		}
	}
}

std::shared_ptr<Program> BytecodeGenerator::generate_executable(std::string filename,
	std::vector<std::string> argv)
{
	ASSERT(m_frame_register_count.size() == 1)
	relocate_labels(m_functions);
	return std::make_shared<Program>(std::move(m_functions), filename, argv);
}

InstructionBlock *BytecodeGenerator::allocate_block(size_t function_id)
{
	ASSERT(function_id < m_functions.size())

	auto function = std::next(m_functions.begin(), function_id);
	auto &new_block = function->blocks.emplace_back();
	return &new_block;
}

std::shared_ptr<Program> BytecodeGenerator::compile(std::shared_ptr<ast::ASTNode> node,
	std::vector<std::string> argv,
	compiler::OptimizationLevel lvl)
{
	auto module = as<ast::Module>(node);
	ASSERT(module)

	if (lvl > compiler::OptimizationLevel::None) { ast::optimizer::constant_folding(node); }

	auto generator = BytecodeGenerator();

	node->codegen(&generator);

	// allocate registers for __main__
	generator.m_functions.front().metadata.register_count = generator.register_count();
	auto executable = generator.generate_executable(module->filename(), argv);
	return executable;
}
}// namespace codegen