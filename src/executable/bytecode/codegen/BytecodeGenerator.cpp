#include "BytecodeGenerator.hpp"

#include "executable/bytecode/instructions/FunctionCall.hpp"
#include "executable/bytecode/instructions/FunctionCallWithKeywords.hpp"
#include "executable/bytecode/instructions/ImportName.hpp"
#include "executable/bytecode/instructions/InplaceAdd.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"
#include "executable/bytecode/instructions/LoadAttr.hpp"
#include "executable/bytecode/instructions/LoadBuildClass.hpp"
#include "executable/bytecode/instructions/LoadMethod.hpp"
#include "executable/bytecode/instructions/LoadName.hpp"
#include "executable/bytecode/instructions/MethodCall.hpp"
#include "executable/bytecode/instructions/ReturnValue.hpp"
#include "executable/bytecode/instructions/StoreAttr.hpp"
#include "executable/bytecode/instructions/StoreName.hpp"
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
			emit<LoadFast>(m_function_id, m_last_register, arg_index, node->ids()[0]);
			return;
		}
	}
	emit<LoadName>(m_function_id, m_last_register, node->ids()[0]);
}


void BytecodeGenerator::visit(const Constant *node)
{
	auto dst_register = allocate_register();
	emit<LoadConst>(m_function_id, dst_register, node->value());
	m_last_register = dst_register;
}

void BytecodeGenerator::visit(const BinaryExpr *node)
{
	auto lhs_register = generate(node->lhs().get(), m_function_id);
	auto rhs_register = generate(node->rhs().get(), m_function_id);
	auto dst_register = allocate_register();

	switch (node->op_type()) {
	case BinaryOpType::PLUS: {
		emit<Add>(m_function_id, dst_register, lhs_register, rhs_register);
	} break;
	case BinaryOpType::MINUS: {
		emit<Subtract>(m_function_id, dst_register, lhs_register, rhs_register);
	} break;
	case BinaryOpType::MULTIPLY: {
		emit<Multiply>(m_function_id, dst_register, lhs_register, rhs_register);
	} break;
	case BinaryOpType::EXP: {
		emit<Exp>(m_function_id, dst_register, lhs_register, rhs_register);
	} break;
	case BinaryOpType::MODULO: {
		emit<Modulo>(m_function_id, dst_register, lhs_register, rhs_register);
	} break;
	case BinaryOpType::SLASH:
		TODO()
	case BinaryOpType::LEFTSHIFT: {
		emit<LeftShift>(m_function_id, dst_register, lhs_register, rhs_register);
	} break;
	case BinaryOpType::RIGHTSHIFT:
		TODO()
	}

	m_last_register = dst_register;
}

void BytecodeGenerator::visit(const FunctionDefinition *node)
{
	m_ctx.push_local_args(node->args());
	auto this_function_info = allocate_function();
	generate(node->args().get(), this_function_info.function_id);

	for (const auto &node : node->body()) { generate(node.get(), this_function_info.function_id); }

	// always return None
	// this can be optimised away later on
	auto none_value_register = allocate_register();
	emit<LoadConst>(
		this_function_info.function_id, none_value_register, NameConstant{ NoneType{} });
	emit<ReturnValue>(this_function_info.function_id, none_value_register);

	std::vector<std::string> arg_names;
	for (const auto &arg_name : node->args()->argument_names()) { arg_names.push_back(arg_name); }

	emit<MakeFunction>(m_function_id, this_function_info.function_id, node->name(), arg_names);

	m_ctx.pop_local_args();
	m_last_register = Register{};
}

void BytecodeGenerator::visit(const Arguments *node)
{
	for (const auto &arg : node->args()) { generate(arg.get(), m_function_id); }

	m_last_register = Register{};
}

void BytecodeGenerator::visit(const Argument *) { m_last_register = Register{}; }

void BytecodeGenerator::visit(const Return *node)
{
	auto src_register = generate(node->value().get(), m_function_id);
	emit<ReturnValue>(m_function_id, src_register);
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
						emit<StoreFast>(m_function_id, arg_index, var, src_register);
						continue;
					}
				}
				emit<StoreName>(m_function_id, var, src_register);
			}
		} else if (auto ast_attr = as<Attribute>(target)) {
			auto dst_register = generate(ast_attr->value().get(), m_function_id);
			emit<StoreAttr>(m_function_id, dst_register, src_register, ast_attr->attr());
		} else if (auto ast_tuple = as<Tuple>(target)) {
			std::vector<Register> dst_registers;
			for (size_t i = 0; i < ast_tuple->elements().size(); ++i) {
				dst_registers.push_back(allocate_register());
			}
			emit<UnpackSequence>(m_function_id, dst_registers, src_register);
			size_t idx{ 0 };
			for (const auto &dst_register : dst_registers) {
				const auto &el = ast_tuple->elements()[idx++];
				emit<StoreName>(m_function_id, as<Name>(el)->ids()[0], dst_register);
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
		keywords.push_back(keyword->arg());
	}

	if (node->function()->node_type() == ASTNodeType::Attribute) {
		auto attr_value = as<Attribute>(node->function())->value();
		auto this_name = as<Name>(attr_value);
		ASSERT(this_name);
		ASSERT(this_name->ids().size() == 1);
		emit<MethodCall>(
			m_function_id, func_register, this_name->ids()[0], std::move(arg_registers));
	} else {
		if (keyword_registers.empty()) {
			emit<FunctionCall>(m_function_id, func_register, std::move(arg_registers));
		} else {
			emit<FunctionCallWithKeywords>(m_function_id,
				func_register,
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
	emit<JumpIfFalse>(m_function_id, test_result_register, orelse_start_label);
	for (const auto &body_statement : node->body()) {
		generate(body_statement.get(), m_function_id);
	}
	emit<Jump>(m_function_id, end_label);

	// else
	bind(orelse_start_label);
	for (const auto &orelse_statement : node->orelse()) {
		generate(orelse_statement.get(), m_function_id);
	}
	bind(end_label);

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
	emit<GetIter>(m_function_id, iterator_register, iterator_func_register);

	// call the __next__ implementation
	auto target_ids = as<Name>(node->target())->ids();
	if (target_ids.size() != 1) { TODO() }
	auto target_name = target_ids[0];

	bind(forloop_start_label);
	emit<ForIter>(
		m_function_id, iter_variable_register, iterator_register, target_name, forloop_end_label);

	generate(node->target().get(), m_function_id);

	// body
	for (const auto &el : node->body()) { generate(el.get(), m_function_id); }
	emit<Jump>(m_function_id, forloop_start_label);

	// orelse
	bind(forloop_end_label);
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
	bind(while_loop_start_label);
	const auto test_result_register = generate(node->test().get(), m_function_id);
	emit<JumpIfFalse>(m_function_id, test_result_register, while_loop_end_label);

	// body
	for (const auto &el : node->body()) { generate(el.get(), m_function_id); }
	emit<Jump>(m_function_id, while_loop_start_label);

	// orelse
	bind(while_loop_end_label);
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
		emit<Equal>(m_function_id, result_reg, lhs_reg, rhs_reg);
	} break;
	case Compare::OpType::LtE: {
		emit<LessThanEquals>(m_function_id, result_reg, lhs_reg, rhs_reg);
	} break;
	case Compare::OpType::Lt: {
		emit<LessThan>(m_function_id, result_reg, lhs_reg, rhs_reg);
	} break;
	default: {
		TODO()
	}
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
	emit<BuildList>(m_function_id, result_reg, element_registers);

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
	emit<BuildTuple>(m_function_id, result_reg, element_registers);

	m_last_register = result_reg;
}

void BytecodeGenerator::visit(const ClassDefinition *node)
{
	if (node->args()) {
		spdlog::error("ClassDefinition cannot handle arguments");
		TODO();
	}

	size_t class_id;
	{
		auto this_class_info = allocate_function();
		class_id = this_class_info.function_id;

		const auto name_register = allocate_register();
		const auto qualname_register = allocate_register();
		const auto return_none_register = allocate_register();

		// class definition preamble, a la CPython
		emit<LoadName>(class_id, name_register, "__name__");
		emit<StoreName>(class_id, "__module__", name_register);
		emit<LoadConst>(class_id, qualname_register, String{ "A" });
		emit<StoreName>(class_id, "__qualname__", qualname_register);

		// the actual class definition
		for (const auto &el : node->body()) { generate(el.get(), class_id); }

		emit<LoadConst>(class_id, return_none_register, NameConstant{ NoneType{} });
		emit<ReturnValue>(class_id, return_none_register);
	}

	const auto builtin_build_class_register = allocate_register();
	const auto class_name_register = allocate_register();
	const auto class_location_register = allocate_register();

	emit<LoadBuildClass>(m_function_id, builtin_build_class_register);
	emit<LoadConst>(m_function_id, class_name_register, String{ node->name() });
	emit<LoadConst>(
		m_function_id, class_location_register, Number{ static_cast<int64_t>(class_id) });

	emit<FunctionCall>(m_function_id,
		builtin_build_class_register,
		std::vector{ class_name_register, class_location_register });

	emit<StoreName>(m_function_id, node->name(), Register{ 0 });

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
	emit<BuildDict>(m_function_id, result_reg, key_registers, value_registers);

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
		emit<LoadMethod>(m_function_id, method_name_register, this_value_register, node->attr());
		m_last_register = method_name_register;
	} else if (node->context() == ContextType::LOAD) {
		auto attribute_value_register = allocate_register();
		emit<LoadAttr>(m_function_id, attribute_value_register, this_value_register, node->attr());
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
		emit<InplaceAdd>(m_function_id, lhs_register, rhs_register);
	} break;
	case BinaryOpType::MINUS: {
		TODO()
	} break;
	case BinaryOpType::MULTIPLY: {
		TODO()
	} break;
	case BinaryOpType::EXP: {
		TODO()
	} break;
	case BinaryOpType::MODULO: {
		TODO()
	} break;
	case BinaryOpType::SLASH:
		TODO()
	case BinaryOpType::LEFTSHIFT: {
		TODO()
	} break;
	case BinaryOpType::RIGHTSHIFT:
		TODO()
	}

	if (auto named_target = as<Name>(node->target())) {
		if (named_target->ids().size() != 1) { TODO() }
		emit<StoreName>(m_function_id, named_target->ids()[0], lhs_register);
	} else {
		TODO()
	}
	m_last_register = lhs_register;
}

void BytecodeGenerator::visit(const Import *node)
{
	auto module_register = allocate_register();
	emit<ImportName>(m_function_id, module_register, node->names());
	if (node->asname().has_value()) {
		emit<StoreName>(m_function_id, *node->asname(), module_register);
	} else {
		emit<StoreName>(m_function_id, node->names()[0], module_register);
	}
	m_last_register = Register{};
}

void BytecodeGenerator::visit(const Module *node)
{
	Register last{ 0 };
	for (const auto &statement : node->body()) { last = generate(statement.get(), m_function_id); }

	emit<ReturnValue>(m_function_id, last);
	m_last_register = Register{ 0 };
}

void BytecodeGenerator::visit(const Subscript *) { TODO() }

void BytecodeGenerator::visit(const Raise *) { TODO() }

void BytecodeGenerator::visit(const Try *) { TODO() }

void BytecodeGenerator::visit(const ExceptHandler *){ TODO() }

FunctionInfo::FunctionInfo(size_t function_id_, BytecodeGenerator *generator_)
	: function_id(function_id_), generator(generator_)
{
	generator->enter_function();
}

FunctionInfo::~FunctionInfo() { generator->exit_function(function_id); }

BytecodeGenerator::BytecodeGenerator() : m_frame_register_count({ start_register })
{
	// allocate main
	allocate_function();
}

BytecodeGenerator::~BytecodeGenerator() {}

void BytecodeGenerator::exit_function(size_t function_id)
{
	m_functions[function_id].metadata.register_count = register_count();
	m_frame_register_count.pop_back();
}

FunctionInfo BytecodeGenerator::allocate_function()
{
	auto &new_func = m_functions.emplace_back();
	new_func.metadata.function_name = std::to_string(m_functions.size() - 1);
	return FunctionInfo{ m_functions.size() - 1, this };
}

void BytecodeGenerator::relocate_labels(const FunctionBlocks &functions)
{
	for (const auto &block : functions) {
		size_t instruction_idx{ 0 };
		for (const auto &ins : block.instructions) { ins->relocate(*this, instruction_idx++); }
	}
}

std::shared_ptr<Program> BytecodeGenerator::generate_executable(std::string filename,
	std::vector<std::string> argv)
{
	ASSERT(m_frame_register_count.size() == 1)
	// make sure that at the end of compiling code we are back to __main__ frame
	// ASSERT(m_frame_register_count.size() == 1)

	// std::vector<std::unique_ptr<Instruction>> instructions;
	// for (const auto &function : m_functions) {
	// 	for (auto &&ins : function.instructions) { instructions.push_back(std::move(ins)); }
	// }


	// FunctionBlocks functions;
	// std::vector<size_t> function_offsets;
	// function_offsets.resize(m_functions.size());
	// size_t offset = 0;
	// spdlog::debug("m_functions size: {}", m_functions.size());
	// for (size_t i = 1; i < m_functions.size(); ++i) {
	// 	auto &new_function_block = functions.emplace_back(std::move(m_functions[i]));
	// 	spdlog::debug("function {} requires {} virtual registers",
	// 		new_function_block.metadata.function_name,
	// 		new_function_block.metadata.register_count);
	// 	new_function_block.metadata.offset = offset;
	// 	function_offsets[i] = offset;
	// 	offset += new_function_block.instructions.size();
	// }

	// auto &main_function_block = functions.emplace_back(std::move(m_functions.front()));
	// main_function_block.metadata.offset = offset;
	// function_offsets.front() = offset;
	// spdlog::debug(
	// 	"__main__ requires {} virtual registers", main_function_block.metadata.register_count);

	relocate_labels(m_functions);

	return std::make_shared<Program>(std::move(m_functions), filename, argv);
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