#include "AST.hpp"
#include "executable/bytecode/BytecodeGenerator.hpp"
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


Register
	Name::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
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


Register
	Constant::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &) const
{
	auto dst_register = generator.allocate_register();
	generator.emit<LoadConst>(function_id, dst_register, m_value);
	return dst_register;
}

Register BinaryExpr::generate_impl(size_t function_id,
	BytecodeGenerator &generator,
	ASTContext &ctx) const
{
	auto lhs_register = m_lhs->generate(function_id, generator, ctx);
	auto rhs_register = m_rhs->generate(function_id, generator, ctx);
	auto dst_register = generator.allocate_register();

	switch (m_op_type) {
	case BinaryOpType::PLUS: {
		generator.emit<Add>(function_id, dst_register, lhs_register, rhs_register);
		return dst_register;
	}
	case BinaryOpType::MINUS: {
		generator.emit<Subtract>(function_id, dst_register, lhs_register, rhs_register);
		return dst_register;
	}
	case BinaryOpType::MULTIPLY: {
		generator.emit<Multiply>(function_id, dst_register, lhs_register, rhs_register);
		return dst_register;
	}
	case BinaryOpType::EXP: {
		generator.emit<Exp>(function_id, dst_register, lhs_register, rhs_register);
		return dst_register;
	}
	case BinaryOpType::MODULO: {
		generator.emit<Modulo>(function_id, dst_register, lhs_register, rhs_register);
		return dst_register;
	}
	case BinaryOpType::SLASH:
		TODO()
	case BinaryOpType::LEFTSHIFT: {
		generator.emit<LeftShift>(function_id, dst_register, lhs_register, rhs_register);
		return dst_register;
	}
	case BinaryOpType::RIGHTSHIFT:
		TODO()
	}
	ASSERT_NOT_REACHED()
}

Register FunctionDefinition::generate_impl(size_t function_id,
	BytecodeGenerator &generator,
	ASTContext &ctx) const
{
	ctx.push_local_args(m_args);
	auto this_function_info = generator.allocate_function();
	m_args->generate(this_function_info.function_id, generator, ctx);

	for (const auto &node : m_body) {
		node->generate(this_function_info.function_id, generator, ctx);
	}

	// always return None
	// this can be optimised away later on
	auto none_value_register = generator.allocate_register();
	generator.emit<LoadConst>(
		this_function_info.function_id, none_value_register, NameConstant{ NoneType{} });
	generator.emit<ReturnValue>(this_function_info.function_id, none_value_register);

	std::vector<std::string> arg_names;
	for (const auto &arg_name : m_args->argument_names()) { arg_names.push_back(arg_name); }

	generator.emit<MakeFunction>(
		function_id, this_function_info.function_id, m_function_name, arg_names);

	ctx.pop_local_args();
	return {};
}

Register Arguments::generate_impl(size_t function_id,
	BytecodeGenerator &generator,
	ASTContext &ctx) const
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

Register Argument::generate_impl(size_t, BytecodeGenerator &, ASTContext &) const { return {}; }

Register
	Return::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	auto src_register = m_value->generate(function_id, generator, ctx);
	generator.emit<ReturnValue>(function_id, src_register);
	// return register
	return 0;
}

Register
	Assign::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	const auto src_register = m_value->generate(function_id, generator, ctx);
	ASSERT(m_targets.size() > 0)

	for (const auto &target : m_targets) {
		if (auto ast_name = as<Name>(target)) {
			for (const auto &var : ast_name->ids()) {
				if (ctx.has_local_args()) {
					const auto &local_args = ctx.local_args();
					const auto &arg_names = local_args->argument_names();
					if (auto it = std::find(arg_names.begin(), arg_names.end(), var);
						it != arg_names.end()) {
						const size_t arg_index = std::distance(arg_names.begin(), it);
						generator.emit<StoreFast>(function_id, arg_index, var, src_register);
						continue;
					}
				}
				generator.emit<StoreName>(function_id, var, src_register);
			}
		} else if (auto ast_attr = as<Attribute>(target)) {
			auto dst_register = ast_attr->value()->generate(function_id, generator, ctx);
			generator.emit<StoreAttr>(function_id, dst_register, src_register, ast_attr->attr());
		} else if (auto ast_tuple = as<Tuple>(target)) {
			std::vector<Register> dst_registers;
			for (size_t i = 0; i < ast_tuple->elements().size(); ++i) {
				dst_registers.push_back(generator.allocate_register());
			}
			generator.emit<UnpackSequence>(function_id, dst_registers, src_register);
			size_t idx{ 0 };
			for (const auto &dst_register : dst_registers) {
				const auto &el = ast_tuple->elements()[idx++];
				generator.emit<StoreName>(function_id, as<Name>(el)->ids()[0], dst_register);
			}
		} else {
			TODO();
		}
	}
	return src_register;
}

Register
	Call::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	std::vector<Register> arg_registers;
	std::vector<Register> keyword_registers;
	std::vector<std::string> keywords;

	arg_registers.reserve(m_args.size());
	keyword_registers.reserve(m_keywords.size());
	keywords.reserve(m_keywords.size());

	auto func_register = m_function->generate(function_id, generator, ctx);

	for (const auto &arg : m_args) {
		arg_registers.push_back(arg->generate(function_id, generator, ctx));
	}

	for (const auto &keyword : m_keywords) {
		keyword_registers.push_back(keyword->generate(function_id, generator, ctx));
		keywords.push_back(keyword->arg());
	}

	if (m_function->node_type() == ASTNodeType::Attribute) {
		auto attr_value = as<Attribute>(m_function)->value();
		auto this_name = as<Name>(attr_value);
		ASSERT(this_name);
		ASSERT(this_name->ids().size() == 1);
		generator.emit<MethodCall>(
			function_id, func_register, this_name->ids()[0], std::move(arg_registers));
	} else {
		if (keyword_registers.empty()) {
			generator.emit<FunctionCall>(function_id, func_register, std::move(arg_registers));
		} else {
			generator.emit<FunctionCallWithKeywords>(function_id,
				func_register,
				std::move(arg_registers),
				std::move(keyword_registers),
				std::move(keywords));
		}
	}

	return {};
}

Register If::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	static size_t if_count = 0;

	ASSERT(!m_body.empty())

	auto orelse_start_label = generator.make_label(fmt::format("ORELSE_{}", if_count), function_id);
	auto end_label = generator.make_label(fmt::format("END_{}", if_count++), function_id);

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

Register For::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	static size_t for_loop_count = 0;

	ASSERT(!m_body.empty())

	auto forloop_start_label =
		generator.make_label(fmt::format("FOR_START_{}", for_loop_count), function_id);
	auto forloop_end_label =
		generator.make_label(fmt::format("FOR_END_{}", for_loop_count++), function_id);

	// generate the iterator
	const auto iterator_func_register = m_iter->generate(function_id, generator, ctx);

	auto iterator_register = generator.allocate_register();
	auto iter_variable_register = generator.allocate_register();

	// call the __iter__ implementation
	generator.emit<GetIter>(function_id, iterator_register, iterator_func_register);

	// call the __next__ implementation
	auto target_ids = as<Name>(m_target)->ids();
	if (target_ids.size() != 1) { TODO() }
	auto target_name = target_ids[0];

	generator.bind(forloop_start_label);
	generator.emit<ForIter>(
		function_id, iter_variable_register, iterator_register, target_name, forloop_end_label);

	m_target->generate(function_id, generator, ctx);

	// body
	for (const auto &el : m_body) { el->generate(function_id, generator, ctx); }
	generator.emit<Jump>(function_id, forloop_start_label);

	// orelse
	generator.bind(forloop_end_label);
	for (const auto &el : m_orelse) { el->generate(function_id, generator, ctx); }

	return {};
}


Register
	While::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	static size_t while_loop_count = 0;

	ASSERT(!m_body.empty())

	auto while_loop_start_label =
		generator.make_label(fmt::format("WHILE_START_{}", while_loop_count), function_id);
	auto while_loop_end_label =
		generator.make_label(fmt::format("WHILE_END_{}", while_loop_count++), function_id);

	// test
	generator.bind(while_loop_start_label);
	const auto test_result_register = m_test->generate(function_id, generator, ctx);
	generator.emit<JumpIfFalse>(function_id, test_result_register, while_loop_end_label);

	// body
	for (const auto &el : m_body) { el->generate(function_id, generator, ctx); }
	generator.emit<Jump>(function_id, while_loop_start_label);

	// orelse
	generator.bind(while_loop_end_label);
	for (const auto &el : m_orelse) { el->generate(function_id, generator, ctx); }

	return {};
}


Register
	Compare::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	const auto lhs_reg = m_lhs->generate(function_id, generator, ctx);
	const auto rhs_reg = m_rhs->generate(function_id, generator, ctx);
	const auto result_reg = generator.allocate_register();

	switch (m_op) {
	case OpType::Eq: {
		generator.emit<Equal>(function_id, result_reg, lhs_reg, rhs_reg);
	} break;
	case OpType::LtE: {
		generator.emit<LessThanEquals>(function_id, result_reg, lhs_reg, rhs_reg);
	} break;
	case OpType::Lt: {
		generator.emit<LessThan>(function_id, result_reg, lhs_reg, rhs_reg);
	} break;
	default: {
		TODO()
	}
	}
	return result_reg;
}


Register
	List::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	std::vector<Register> element_registers;
	element_registers.reserve(m_elements.size());

	for (const auto &el : m_elements) {
		element_registers.push_back(el->generate(function_id, generator, ctx));
	}

	const auto result_reg = generator.allocate_register();
	generator.emit<BuildList>(function_id, result_reg, element_registers);

	return result_reg;
}


Register
	Tuple::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	std::vector<Register> element_registers;
	element_registers.reserve(m_elements.size());

	for (const auto &el : m_elements) {
		element_registers.push_back(el->generate(function_id, generator, ctx));
	}

	const auto result_reg = generator.allocate_register();
	generator.emit<BuildTuple>(function_id, result_reg, element_registers);

	return result_reg;
}


Register ClassDefinition::generate_impl(size_t function_id,
	BytecodeGenerator &generator,
	ASTContext &ctx) const
{
	if (m_args) {
		spdlog::error("ClassDefinition cannot handle arguments");
		TODO();
	}

	size_t class_id;
	{
		auto this_class_info = generator.allocate_function();
		class_id = this_class_info.function_id;

		auto name_register = generator.allocate_register();
		auto qualname_register = generator.allocate_register();
		auto return_none_register = generator.allocate_register();

		// class definition preamble, a la CPython
		generator.emit<LoadName>(class_id, name_register, "__name__");
		generator.emit<StoreName>(class_id, "__module__", name_register);
		generator.emit<LoadConst>(class_id, qualname_register, String{ "A" });
		generator.emit<StoreName>(class_id, "__qualname__", qualname_register);

		// the actual class definition
		for (const auto &el : m_body) { el->generate(class_id, generator, ctx); }

		generator.emit<LoadConst>(class_id, return_none_register, NameConstant{ NoneType{} });
		generator.emit<ReturnValue>(class_id, return_none_register);
	}

	auto builtin_build_class_register = generator.allocate_register();
	auto class_name_register = generator.allocate_register();
	auto class_location_register = generator.allocate_register();

	generator.emit<LoadBuildClass>(function_id, builtin_build_class_register);
	generator.emit<LoadConst>(function_id, class_name_register, String{ m_class_name });
	generator.emit<LoadConst>(
		function_id, class_location_register, Number{ static_cast<int64_t>(class_id) });

	generator.emit<FunctionCall>(function_id,
		builtin_build_class_register,
		std::vector{ class_name_register, class_location_register });

	generator.emit<StoreName>(function_id, m_class_name, Register{ 0 });

	return {};
}


Register
	Dict::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	ASSERT(m_keys.size() == m_values.size())

	std::vector<Register> key_registers;
	std::vector<Register> value_registers;

	for (const auto &key : m_keys) {
		key_registers.push_back(key->generate(function_id, generator, ctx));
	}
	for (const auto &value : m_values) {
		value_registers.push_back(value->generate(function_id, generator, ctx));
	}

	const auto result_reg = generator.allocate_register();
	generator.emit<BuildDict>(function_id, result_reg, key_registers, value_registers);

	return result_reg;
}


Register Attribute::generate_impl(size_t function_id,
	BytecodeGenerator &generator,
	ASTContext &ctx) const
{
	auto this_value_register = m_value->generate(function_id, generator, ctx);

	const auto *parent_node = ctx.parent_nodes()[ctx.parent_nodes().size() - 2];
	auto parent_node_type = parent_node->node_type();

	// the parent is a Call AST node and this is the function that is being called, then this
	// must be a method "foo.bar()" -> .bar() is the function being called by parent AST node
	// and this attribute
	if (parent_node_type == ASTNodeType::Call
		&& static_cast<const Call *>(parent_node)->function().get() == this) {
		auto method_name_register = generator.allocate_register();
		generator.emit<LoadMethod>(function_id, method_name_register, this_value_register, m_attr);
		return method_name_register;
	}

	if (m_ctx == ContextType::LOAD) {
		auto attribute_value_register = generator.allocate_register();
		generator.emit<LoadAttr>(
			function_id, attribute_value_register, this_value_register, m_attr);
		return attribute_value_register;
	}

	TODO();
}


Register
	Keyword::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &ctx) const
{
	return m_value->generate(function_id, generator, ctx);
}


Register AugAssign::generate_impl(size_t function_id,
	BytecodeGenerator &generator,
	ASTContext &ctx) const
{
	const auto lhs_register = m_target->generate(function_id, generator, ctx);
	const auto rhs_register = m_value->generate(function_id, generator, ctx);
	switch (m_op) {
	case BinaryOpType::PLUS: {
		generator.emit<InplaceAdd>(function_id, lhs_register, rhs_register);
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

	if (auto named_target = as<Name>(m_target)) {
		if (named_target->ids().size() != 1) { TODO() }
		generator.emit<StoreName>(function_id, named_target->ids()[0], lhs_register);
	} else {
		TODO()
	}
	return lhs_register;
}


Register Import::generate_impl(size_t function_id, BytecodeGenerator &generator, ASTContext &) const
{
	auto module_register = generator.allocate_register();
	generator.emit<ImportName>(function_id, module_register, m_names);
	if (m_asname.has_value()) {
		generator.emit<StoreName>(function_id, *m_asname, module_register);
	} else {
		generator.emit<StoreName>(function_id, m_names[0], module_register);
	}
	return {};
}


#define __AST_NODE_TYPE(NodeType) \
	void NodeType::generate_(CodeGenerator *generator) const { generator->visit(this); }
AST_NODE_TYPES
#undef __AST_NODE_TYPE


}// namespace ast