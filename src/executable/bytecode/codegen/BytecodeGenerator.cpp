#include "BytecodeGenerator.hpp"
#include "executable/bytecode/BytecodeProgram.hpp"
#include "executable/bytecode/instructions/ClearExceptionState.hpp"
#include "executable/bytecode/instructions/DictMerge.hpp"
#include "executable/bytecode/instructions/FunctionCall.hpp"
#include "executable/bytecode/instructions/FunctionCallEx.hpp"
#include "executable/bytecode/instructions/FunctionCallWithKeywords.hpp"
#include "executable/bytecode/instructions/GreaterThan.hpp"
#include "executable/bytecode/instructions/GreaterThanEquals.hpp"
#include "executable/bytecode/instructions/ImportName.hpp"
#include "executable/bytecode/instructions/InOp.hpp"
#include "executable/bytecode/instructions/InplaceAdd.hpp"
#include "executable/bytecode/instructions/InplaceSub.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"
#include "executable/bytecode/instructions/IsOp.hpp"
#include "executable/bytecode/instructions/JumpForward.hpp"
#include "executable/bytecode/instructions/JumpIfFalseOrPop.hpp"
#include "executable/bytecode/instructions/JumpIfNotExceptionMatch.hpp"
#include "executable/bytecode/instructions/JumpIfTrue.hpp"
#include "executable/bytecode/instructions/JumpIfTrueOrPop.hpp"
#include "executable/bytecode/instructions/ListExtend.hpp"
#include "executable/bytecode/instructions/ListToTuple.hpp"
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
#include "executable/bytecode/instructions/TrueDivide.cpp"
#include "executable/bytecode/instructions/Unary.hpp"
#include "executable/bytecode/instructions/UnpackSequence.hpp"


#include "ast/optimizers/ConstantFolding.hpp"
#include "executable/FunctionBlock.hpp"
#include "executable/Mangler.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"

#include <filesystem>

namespace fs = std::filesystem;

using namespace ast;

namespace codegen {

Value *BytecodeGenerator::visit(const Name *node)
{
	auto *dst = create_value();

	if (m_ctx.has_local_args()) {
		const auto &local_args = m_ctx.local_args();
		const auto &arg_names = local_args->argument_names();
		const auto &kw_only_arg_names = local_args->kw_only_argument_names();
		if (auto it = std::find(arg_names.begin(), arg_names.end(), node->ids()[0]);
			it != arg_names.end()) {
			const size_t arg_index = std::distance(arg_names.begin(), it);
			emit<LoadFast>(dst->get_register(), arg_index, node->ids()[0]);
			return dst;
		} else if (auto it = std::find(
					   kw_only_arg_names.begin(), kw_only_arg_names.end(), node->ids()[0]);
				   it != kw_only_arg_names.end()) {
			const size_t arg_index =
				std::distance(kw_only_arg_names.begin(), it) + arg_names.size();
			emit<LoadFast>(dst->get_register(), arg_index, node->ids()[0]);
			return dst;
		} else if (local_args->vararg() && local_args->vararg()->name() == node->ids()[0]) {
			const size_t arg_index = local_args->args().size() + local_args->kwonlyargs().size();
			emit<LoadFast>(dst->get_register(), arg_index, node->ids()[0]);
			return dst;
		} else if (local_args->kwarg() && local_args->kwarg()->name() == node->ids()[0]) {
			size_t arg_index = local_args->args().size() + local_args->kwonlyargs().size();
			if (local_args->vararg()) { arg_index++; }
			emit<LoadFast>(dst->get_register(), arg_index, node->ids()[0]);
			return dst;
		}
	}
	emit<LoadName>(dst->get_register(), node->ids()[0]);
	return dst;
}


Value *BytecodeGenerator::visit(const Constant *node)
{
	auto *dst = create_value();
	emit<LoadConst>(dst->get_register(), *node->value());
	return dst;
}

Value *BytecodeGenerator::visit(const BinaryExpr *node)
{
	auto *lhs = generate(node->lhs().get(), m_function_id);
	auto *rhs = generate(node->rhs().get(), m_function_id);
	auto *dst = create_value();

	switch (node->op_type()) {
	case BinaryOpType::PLUS: {
		emit<Add>(dst->get_register(), lhs->get_register(), rhs->get_register());
	} break;
	case BinaryOpType::MINUS: {
		emit<Subtract>(dst->get_register(), lhs->get_register(), rhs->get_register());
	} break;
	case BinaryOpType::MULTIPLY: {
		emit<Multiply>(dst->get_register(), lhs->get_register(), rhs->get_register());
	} break;
	case BinaryOpType::EXP: {
		emit<Exp>(dst->get_register(), lhs->get_register(), rhs->get_register());
	} break;
	case BinaryOpType::MODULO: {
		emit<Modulo>(dst->get_register(), lhs->get_register(), rhs->get_register());
	} break;
	case BinaryOpType::SLASH: {
		emit<TrueDivide>(dst->get_register(), lhs->get_register(), rhs->get_register());
	} break;
	case BinaryOpType::FLOORDIV:
		TODO();
	case BinaryOpType::LEFTSHIFT: {
		emit<LeftShift>(dst->get_register(), lhs->get_register(), rhs->get_register());
	} break;
	case BinaryOpType::RIGHTSHIFT:
		TODO();
	}

	return dst;
}

Value *BytecodeGenerator::visit(const FunctionDefinition *node)
{
	if (!node->decorator_list().empty()) { TODO(); }

	std::vector<std::string> arg_names;
	m_ctx.push_local_args(node->args());
	const std::string &function_name = Mangler::default_mangler().function_mangle(
		mangle_namespace(m_stack), node->name(), node->source_location());
	auto *f = create_function(function_name);

	m_stack.push(Scope{ .name = node->name() });

	auto *block = allocate_block(f->function_info().function_id);
	auto *old_block = m_current_block;
	set_insert_point(block);

	generate(node->args().get(), f->function_info().function_id);

	for (const auto &node : node->body()) { generate(node.get(), f->function_info().function_id); }

	// always return None
	// this can be optimised away later on
	auto none_value_register = allocate_register();
	emit<LoadConst>(none_value_register, py::NameConstant{ py::NoneType{} });
	emit<ReturnValue>(none_value_register);

	for (const auto &arg_name : node->args()->argument_names()) { arg_names.push_back(arg_name); }
	for (const auto &arg_name : node->args()->kw_only_argument_names()) {
		arg_names.push_back(arg_name);
	}

	set_insert_point(old_block);
	m_ctx.pop_local_args();
	m_stack.pop();

	size_t arg_count = node->args()->args().size();
	size_t kwonly_arg_count = node->args()->kwonlyargs().size();

	std::vector<Register> defaults;
	defaults.reserve(node->args()->defaults().size());
	for (const auto &default_node : node->args()->defaults()) {
		defaults.push_back(generate(default_node.get(), m_function_id)->get_register());
	}

	std::vector<std::optional<Register>> kw_defaults;
	kw_defaults.reserve(node->args()->kw_defaults().size());
	for (const auto &default_node : node->args()->kw_defaults()) {
		if (default_node) {
			kw_defaults.push_back(generate(default_node.get(), m_function_id)->get_register());
		} else {
			kw_defaults.push_back(std::nullopt);
		}
	}

	emit<MakeFunction>(f->get_name(),
		arg_names,
		defaults,
		kw_defaults,
		arg_count,
		kwonly_arg_count,
		node->args()->vararg() != nullptr,
		node->args()->kwarg() != nullptr);

	exit_function(f->function_info().function_id);

	return f;
}

Value *BytecodeGenerator::visit(const Arguments *node)
{
	// if (!node->kw_defaults().empty()) { TODO(); }

	for (const auto &arg : node->args()) { generate(arg.get(), m_function_id); }
	if (node->vararg()) { generate(node->vararg().get(), m_function_id); }
	if (node->kwarg()) { generate(node->kwarg().get(), m_function_id); }
	for (const auto &arg : node->kwonlyargs()) { generate(arg.get(), m_function_id); }

	return nullptr;
}

Value *BytecodeGenerator::visit(const Argument *) { return nullptr; }

Value *BytecodeGenerator::visit(const Starred *node)
{
	if (node->ctx() != ContextType::LOAD) { TODO(); }
	return generate(node->value().get(), m_function_id);
}

Value *BytecodeGenerator::visit(const Return *node)
{
	auto *src = generate(node->value().get(), m_function_id);
	emit<ReturnValue>(src->get_register());
	return src;
}

Value *BytecodeGenerator::visit(const Assign *node)
{
	auto *src = generate(node->value().get(), m_function_id);
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
						emit<StoreFast>(arg_index, var, src->get_register());
						continue;
					}
				}
				emit<StoreName>(var, src->get_register());
			}
		} else if (auto ast_attr = as<Attribute>(target)) {
			auto *dst = generate(ast_attr->value().get(), m_function_id);
			emit<StoreAttr>(dst->get_register(), src->get_register(), ast_attr->attr());
		} else if (auto ast_tuple = as<Tuple>(target)) {
			std::vector<Register> dst_registers;
			for (size_t i = 0; i < ast_tuple->elements().size(); ++i) {
				dst_registers.push_back(allocate_register());
			}
			emit<UnpackSequence>(dst_registers, src->get_register());
			size_t idx{ 0 };
			for (const auto &dst_register : dst_registers) {
				const auto &el = ast_tuple->elements()[idx++];
				emit<StoreName>(as<Name>(el)->ids()[0], dst_register);
			}
		} else {
			TODO();
		}
	}
	return src;
}

Value *BytecodeGenerator::visit(const Call *node)
{
	std::vector<BytecodeValue *> arg_values;
	std::vector<BytecodeValue *> keyword_values;
	std::vector<std::string> keywords;

	auto *func = generate(node->function().get(), m_function_id);

	auto is_args_expansion = [](const std::shared_ptr<ASTNode> &node) {
		if (node->node_type() == ASTNodeType::Starred) { return true; }
		return false;
	};

	auto is_kwargs_expansion = [](const std::shared_ptr<Keyword> &node) {
		if (!node->arg().has_value()) { return true; }
		return false;
	};

	bool requires_args_expansion =
		std::any_of(node->args().begin(), node->args().end(), is_args_expansion);
	bool requires_kwargs_expansion =
		std::any_of(node->keywords().begin(), node->keywords().end(), is_kwargs_expansion);

	if (requires_args_expansion || requires_kwargs_expansion) {
		auto *list_value = create_value();
		bool first_args_expansion = true;
		std::vector<Register> args_lhs;
		for (const auto &arg : node->args()) {
			if (is_args_expansion(arg)) {
				if (first_args_expansion) {
					emit<BuildList>(list_value->get_register(), args_lhs);
					args_lhs.clear();
					first_args_expansion = false;
				}
				auto arg_value = generate(arg.get(), m_function_id);
				emit<ListExtend>(list_value->get_register(), arg_value->get_register());
			} else {
				auto *arg_value = generate(arg.get(), m_function_id);
				if (first_args_expansion) {
					args_lhs.push_back(arg_value->get_register());
				} else {
					emit<ListExtend>(list_value->get_register(), arg_value->get_register());
				}
			}
		}
		// we didn't hit any *args, but we still want to build a list of args so that
		// we can call FunctionCallEx
		if (first_args_expansion) { emit<BuildList>(list_value->get_register(), args_lhs); }
		auto *args_tuple = create_value();
		emit<ListToTuple>(args_tuple->get_register(), list_value->get_register());
		arg_values.push_back(args_tuple);

		if (requires_kwargs_expansion) {
			auto *dict_value = create_value();
			std::vector<Register> key_registers;
			std::vector<Register> value_registers;
			bool first_kwargs_expansion = true;

			for (const auto &el : node->keywords()) {
				if (is_kwargs_expansion(el)) {
					if (first_kwargs_expansion) {
						emit<BuildDict>(dict_value->get_register(), key_registers, value_registers);
						value_registers.clear();
						key_registers.clear();
						first_kwargs_expansion = false;
					}
					auto *kwargs_dict = generate(el->value().get(), m_function_id);
					emit<DictMerge>(dict_value->get_register(), kwargs_dict->get_register());
				} else {
					auto *value = generate(el.get(), m_function_id);
					const auto &name = *el->arg();
					auto *key = create_value();
					emit<LoadConst>(key->get_register(), py::String{ name });
					if (first_kwargs_expansion) {
						key_registers.push_back(key->get_register());
						value_registers.push_back(value->get_register());
					} else {
						const auto new_dict_reg = allocate_register();
						emit<BuildDict>(new_dict_reg,
							std::vector<Register>{ key->get_register() },
							std::vector<Register>{ value->get_register() });
						emit<DictMerge>(dict_value->get_register(), new_dict_reg);
					}
				}
			}
			ASSERT(first_kwargs_expansion == false)
			keyword_values.push_back(dict_value);
		} else {
			// dummy value that will be ignore at runtime, since requires_kwargs_expansion is false
			keyword_values.push_back(nullptr);
		}
	} else {
		arg_values.reserve(node->args().size());
		for (const auto &arg : node->args()) {
			arg_values.push_back(generate(arg.get(), m_function_id));
		}
		keyword_values.reserve(node->keywords().size());
		keywords.reserve(node->keywords().size());

		for (const auto &keyword : node->keywords()) {
			keyword_values.push_back(generate(keyword.get(), m_function_id));
			auto keyword_argname = keyword->arg();
			if (!keyword_argname.has_value()) { TODO(); }
			keywords.push_back(*keyword_argname);
		}
	}

	if (requires_args_expansion || requires_kwargs_expansion) {
		ASSERT(arg_values.size() == 1)
		ASSERT(keyword_values.size() == 1)

		emit<FunctionCallEx>(func->get_register(),
			arg_values[0]->get_register(),
			keyword_values[0] ? keyword_values[0]->get_register() : Register{ 0 },
			requires_args_expansion,
			requires_kwargs_expansion);
	} else {
		if (node->function()->node_type() == ASTNodeType::Attribute) {
			auto attr_name = as<Attribute>(node->function())->attr();
			std::vector<Register> arg_registers;
			arg_registers.reserve(arg_values.size());
			for (const auto &arg : arg_values) { arg_registers.push_back(arg->get_register()); }
			emit<MethodCall>(func->get_register(), attr_name, std::move(arg_registers));
		} else {
			std::vector<Register> arg_registers;
			std::vector<Register> keyword_registers;

			arg_registers.reserve(arg_values.size());
			for (const auto &arg : arg_values) { arg_registers.push_back(arg->get_register()); }

			keyword_registers.reserve(keyword_values.size());
			for (const auto &kw : keyword_values) {
				keyword_registers.push_back(kw->get_register());
			}

			if (keyword_registers.empty()) {
				emit<FunctionCall>(func->get_register(), std::move(arg_registers));
			} else {
				emit<FunctionCallWithKeywords>(func->get_register(),
					std::move(arg_registers),
					std::move(keyword_registers),
					std::move(keywords));
			}
		}
	}

	return create_return_value();
}

Value *BytecodeGenerator::visit(const If *node)
{
	static size_t if_count = 0;

	ASSERT(!node->body().empty())

	auto orelse_start_label = make_label(fmt::format("ORELSE_{}", if_count), m_function_id);
	auto end_label = make_label(fmt::format("END_{}", if_count++), m_function_id);

	// if
	auto *test_result = generate(node->test().get(), m_function_id);
	emit<JumpIfFalse>(test_result->get_register(), orelse_start_label);
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

	return nullptr;
}

Value *BytecodeGenerator::visit(const For *node)
{
	static size_t for_loop_count = 0;

	ASSERT(!node->body().empty())

	auto forloop_start_label =
		make_label(fmt::format("FOR_START_{}", for_loop_count), m_function_id);
	auto forloop_end_label = make_label(fmt::format("FOR_END_{}", for_loop_count++), m_function_id);

	// generate the iterator
	auto *iterator_func = generate(node->iter().get(), m_function_id);

	auto iterator_register = allocate_register();
	auto iter_variable_register = allocate_register();

	// call the __iter__ implementation
	emit<GetIter>(iterator_register, iterator_func->get_register());

	// call the __next__ implementation
	auto target_ids = as<Name>(node->target())->ids();
	if (target_ids.size() != 1) { TODO(); }
	auto target_name = target_ids[0];

	bind(*forloop_start_label);
	emit<ForIter>(iter_variable_register, iterator_register, forloop_end_label);
	emit<StoreName>(target_name, iter_variable_register);

	generate(node->target().get(), m_function_id);

	// body
	for (const auto &el : node->body()) { generate(el.get(), m_function_id); }
	emit<Jump>(forloop_start_label);

	// orelse
	bind(*forloop_end_label);
	for (const auto &el : node->orelse()) { generate(el.get(), m_function_id); }

	return nullptr;
}

Value *BytecodeGenerator::visit(const While *node)
{
	static size_t while_loop_count = 0;

	ASSERT(!node->body().empty())

	auto while_loop_start_label =
		make_label(fmt::format("WHILE_START_{}", while_loop_count), m_function_id);
	auto while_loop_end_label =
		make_label(fmt::format("WHILE_END_{}", while_loop_count++), m_function_id);

	// test
	bind(*while_loop_start_label);
	const auto *test_result = generate(node->test().get(), m_function_id);
	emit<JumpIfFalse>(test_result->get_register(), while_loop_end_label);

	// body
	for (const auto &el : node->body()) { generate(el.get(), m_function_id); }
	emit<Jump>(while_loop_start_label);

	// orelse
	bind(*while_loop_end_label);
	for (const auto &el : node->orelse()) { generate(el.get(), m_function_id); }

	return nullptr;
}

Value *BytecodeGenerator::visit(const Compare *node)
{
	const auto *lhs = generate(node->lhs().get(), m_function_id);
	const auto *rhs = generate(node->rhs().get(), m_function_id);
	auto *result = create_value();

	switch (node->op()) {
	case Compare::OpType::Eq: {
		emit<Equal>(result->get_register(), lhs->get_register(), rhs->get_register());
	} break;
	case Compare::OpType::NotEq: {
		emit<NotEqual>(result->get_register(), lhs->get_register(), rhs->get_register());
	} break;
	case Compare::OpType::Lt: {
		emit<LessThan>(result->get_register(), lhs->get_register(), rhs->get_register());
	} break;
	case Compare::OpType::LtE: {
		emit<LessThanEquals>(result->get_register(), lhs->get_register(), rhs->get_register());
	} break;
	case Compare::OpType::Gt: {
		emit<GreaterThan>(result->get_register(), lhs->get_register(), rhs->get_register());
	} break;
	case Compare::OpType::GtE: {
		emit<GreaterThanEquals>(result->get_register(), lhs->get_register(), rhs->get_register());
	} break;
	case Compare::OpType::Is: {
		emit<IsOp>(result->get_register(), lhs->get_register(), rhs->get_register(), false);
	} break;
	case Compare::OpType::IsNot: {
		emit<IsOp>(result->get_register(), lhs->get_register(), rhs->get_register(), true);
	} break;
	case Compare::OpType::In: {
		emit<InOp>(result->get_register(), lhs->get_register(), rhs->get_register(), false);
	} break;
	case Compare::OpType::NotIn: {
		emit<InOp>(result->get_register(), lhs->get_register(), rhs->get_register(), true);
	} break;
	}

	return result;
}

Value *BytecodeGenerator::visit(const List *node)
{
	std::vector<Register> element_registers;
	element_registers.reserve(node->elements().size());

	for (const auto &el : node->elements()) {
		auto *element_value = generate(el.get(), m_function_id);
		element_registers.push_back(element_value->get_register());
	}

	auto *result = create_value();
	emit<BuildList>(result->get_register(), element_registers);

	return result;
}

Value *BytecodeGenerator::visit(const Tuple *node)
{
	std::vector<Register> element_registers;
	element_registers.reserve(node->elements().size());

	for (const auto &el : node->elements()) {
		auto *element_value = generate(el.get(), m_function_id);
		element_registers.push_back(element_value->get_register());
	}

	auto *result = create_value();
	emit<BuildTuple>(result->get_register(), element_registers);

	return result;
}

Value *BytecodeGenerator::visit(const ClassDefinition *node)
{
	if (!node->decorator_list().empty()) { TODO(); }
	std::string class_mangled_name;
	size_t class_id;
	{
		class_mangled_name = Mangler::default_mangler().class_mangle(
			mangle_namespace(m_stack), node->name(), node->source_location());

		auto *class_builder_func = create_function(class_mangled_name);
		m_stack.push(Scope{ .name = node->name() });
		class_id = class_builder_func->function_info().function_id;

		auto *block = allocate_block(class_id);
		auto *old_block = m_current_block;
		set_insert_point(block);

		const auto name_register = allocate_register();
		const auto qualname_register = allocate_register();
		const auto return_none_register = allocate_register();

		// class definition preamble, a la CPython
		emit<LoadName>(name_register, "__name__");
		emit<StoreName>("__module__", name_register);
		emit<LoadConst>(qualname_register, py::String{ node->name() });
		emit<StoreName>("__qualname__", qualname_register);

		// the actual class definition
		for (const auto &el : node->body()) { generate(el.get(), class_id); }

		emit<LoadConst>(return_none_register, py::NameConstant{ py::NoneType{} });
		emit<ReturnValue>(return_none_register);
		m_stack.pop();
		exit_function(class_builder_func->function_info().function_id);

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
		auto *base_value = generate(base.get(), m_function_id);
		arg_registers.push_back(base_value->get_register());
	}
	for (const auto &keyword : node->keywords()) {
		auto *kw_value = generate(keyword.get(), m_function_id);
		kwarg_registers.push_back(kw_value->get_register());
		if (!keyword->arg().has_value()) { TODO(); }
		keyword_names.push_back(*keyword->arg());
	}

	emit<LoadBuildClass>(builtin_build_class_register);
	emit<LoadConst>(class_name_register, py::String{ class_mangled_name });
	emit<LoadConst>(class_location_register, py::Number{ static_cast<int64_t>(class_id) });

	if (kwarg_registers.empty()) {
		emit<FunctionCall>(builtin_build_class_register, std::move(arg_registers));
	} else {
		emit<FunctionCallWithKeywords>(builtin_build_class_register,
			std::move(arg_registers),
			std::move(kwarg_registers),
			std::move(keyword_names));
	}

	emit<StoreName>(node->name(), Register{ 0 });

	return nullptr;
}

Value *BytecodeGenerator::visit(const Dict *node)
{
	ASSERT(node->keys().size() == node->values().size())

	std::vector<Register> key_registers;
	std::vector<Register> value_registers;

	for (const auto &key : node->keys()) {
		auto *key_value = generate(key.get(), m_function_id);
		key_registers.push_back(key_value->get_register());
	}
	for (const auto &value : node->values()) {
		auto *v = generate(value.get(), m_function_id);
		value_registers.push_back(v->get_register());
	}

	auto *result = create_value();
	emit<BuildDict>(result->get_register(), key_registers, value_registers);

	return result;
}

Value *BytecodeGenerator::visit(const Attribute *node)
{
	auto *this_value = generate(node->value().get(), m_function_id);

	const auto *parent_node = m_ctx.parent_nodes()[m_ctx.parent_nodes().size() - 2];
	auto parent_node_type = parent_node->node_type();

	// the parent is a Call AST node and this is the function that is being called, then this
	// must be a method "foo.bar()" -> .bar() is the function being called by parent AST node
	// and this attribute
	if (parent_node_type == ASTNodeType::Call
		&& static_cast<const Call *>(parent_node)->function().get() == node) {
		auto method_name = create_value();
		emit<LoadMethod>(method_name->get_register(), this_value->get_register(), node->attr());
		return method_name;
	} else if (node->context() == ContextType::LOAD) {
		auto *attribute_value = create_value();
		emit<LoadAttr>(attribute_value->get_register(), this_value->get_register(), node->attr());
		return attribute_value;
	} else if (node->context() == ContextType::STORE) {
		auto *attribute_value = create_value();
		emit<LoadAttr>(attribute_value->get_register(), this_value->get_register(), node->attr());
		return attribute_value;
	} else {
		TODO();
	}
}

Value *BytecodeGenerator::visit(const Keyword *node)
{
	return generate(node->value().get(), m_function_id);
}

Value *BytecodeGenerator::visit(const AugAssign *node)
{
	auto *lhs = generate(node->target().get(), m_function_id);
	const auto *rhs = generate(node->value().get(), m_function_id);
	switch (node->op()) {
	case BinaryOpType::PLUS: {
		emit<InplaceAdd>(lhs->get_register(), rhs->get_register());
	} break;
	case BinaryOpType::MINUS: {
		emit<InplaceSub>(lhs->get_register(), rhs->get_register());
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
		emit<StoreName>(named_target->ids()[0], lhs->get_register());
	} else if (auto attr = as<Attribute>(node->target())) {
		auto *obj = generate(attr->value().get(), m_function_id);
		emit<StoreAttr>(obj->get_register(), lhs->get_register(), attr->attr());
	} else {
		TODO();
	}

	return lhs;
}

Value *BytecodeGenerator::visit(const Import *node)
{
	auto module_register = allocate_register();
	emit<ImportName>(module_register, node->names());
	if (node->asname().has_value()) {
		emit<StoreName>(*node->asname(), module_register);
	} else {
		emit<StoreName>(node->names()[0], module_register);
	}
	return nullptr;
}

Value *BytecodeGenerator::visit(const Module *node)
{
	const auto &module_name = fs::path(node->filename()).stem();
	m_stack.push(Scope{ .name = module_name });
	BytecodeValue *last = nullptr;
	for (const auto &statement : node->body()) { last = generate(statement.get(), m_function_id); }

	// TODO: should the module return the last value if there is one?
	last = create_value();
	emit<LoadConst>(last->get_register(), py::NameConstant{ py::NoneType{} });
	emit<ReturnValue>(last->get_register());
	m_stack.pop();
	return last;
}

Value *BytecodeGenerator::visit(const Subscript *) { TODO(); }

Value *BytecodeGenerator::visit(const Raise *) { TODO(); }

Value *BytecodeGenerator::visit(const With *) { TODO(); }

Value *BytecodeGenerator::visit(const WithItem *node)
{
	auto *ctx_expr = generate(node->context_expr().get(), m_function_id);

	if (node->optional_vars()) {
		ASSERT(as<Name>(node->context_expr()))
		ASSERT(as<Name>(node->context_expr())->ids().size() == 1)
		emit<StoreName>(as<Name>(node->context_expr())->ids()[0], ctx_expr->get_register());
	}

	return ctx_expr;
}

Value *BytecodeGenerator::visit(const IfExpr *) { TODO(); }

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

Value *BytecodeGenerator::visit(const Try *node)
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
		if (!handler->type()) {
			// TODO: implement exception handling that catches all exceptions
			TODO();
		}
		auto *exception_type = generate(handler->type().get(), m_function_id);
		emit<JumpIfNotExceptionMatch>(exception_type->get_register());
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

	return nullptr;
}

Value *BytecodeGenerator::visit(const ExceptHandler *) { TODO(); }

Value *BytecodeGenerator::visit(const Global *) { TODO(); }

Value *BytecodeGenerator::visit(const Delete *) { TODO(); }

Value *BytecodeGenerator::visit(const UnaryExpr *node)
{
	const auto *src = generate(node->operand().get(), m_function_id);
	auto *dst = create_value();
	switch (node->op_type()) {
	case UnaryOpType::ADD: {
		emit<UnaryPositive>(dst->get_register(), src->get_register());
	} break;
	case UnaryOpType::SUB: {
		emit<UnaryNegative>(dst->get_register(), src->get_register());
	} break;
	case UnaryOpType::INVERT: {
		emit<UnaryInvert>(dst->get_register(), src->get_register());
	} break;
	case UnaryOpType::NOT: {
		emit<UnaryNot>(dst->get_register(), src->get_register());
	} break;
	}

	return dst;
}

Value *BytecodeGenerator::visit(const BoolOp *node)
{
	static size_t bool_op_count = 0;
	auto end_label = make_label(fmt::format("BOOL_OP_END_{}", bool_op_count++), m_function_id);
	auto *result = create_value();
	auto *last_result = create_value();
	switch (node->op()) {
	case BoolOp::OpType::And: {
		auto it = node->values().begin();
		auto end = node->values().end();
		while (std::next(it) != end) {
			last_result = generate((*it).get(), m_function_id);
			emit<JumpIfFalseOrPop>(last_result->get_register(), result->get_register(), end_label);
			it++;
		}
		last_result = generate((*it).get(), m_function_id);
	} break;
	case BoolOp::OpType::Or: {
		auto it = node->values().begin();
		auto end = node->values().end();
		while (std::next(it) != end) {
			last_result = generate((*it).get(), m_function_id);
			emit<JumpIfTrueOrPop>(last_result->get_register(), result->get_register(), end_label);
			it++;
		}
		last_result = generate((*it).get(), m_function_id);
	}
	}
	emit<Move>(result->get_register(), last_result->get_register());

	bind(*end_label);
	return result;
}

Value *BytecodeGenerator::visit(const Assert *node)
{
	static size_t assert_count = 0;
	auto end_label = make_label(fmt::format("END_{}", assert_count++), m_function_id);

	auto *test_result = generate(node->test().get(), m_function_id);

	emit<JumpIfTrue>(test_result->get_register(), end_label);

	auto *assertion_function = create_value();
	emit<LoadAssertionError>(assertion_function->get_register());

	std::vector<Register> args;
	if (node->msg()) { args.push_back(generate(node->msg().get(), m_function_id)->get_register()); }

	emit<FunctionCall>(assertion_function->get_register(), std::move(args));
	auto *exception = create_return_value();
	emit<RaiseVarargs>(exception->get_register());
	bind(*end_label);

	return nullptr;
}

Value *BytecodeGenerator::visit(const Pass *) { return nullptr; }

Value *BytecodeGenerator::visit(const NamedExpr *) { TODO(); }

FunctionInfo::FunctionInfo(size_t function_id_, FunctionBlock &f, BytecodeGenerator *generator_)
	: function_id(function_id_), function(f), generator(generator_)
{
	generator->enter_function();
}

BytecodeGenerator::BytecodeGenerator()
{
	(void)create_function("__main__entry__");
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

BytecodeFunctionValue *BytecodeGenerator::create_function(const std::string &name)
{
	// auto &new_func = m_functions.emplace_back();
	// // allocate the first block
	// new_func.blocks.emplace_back();
	// new_func.metadata.function_name = std::to_string(m_functions.size() - 1);
	// return FunctionInfo{ m_functions.size() - 1, new_func, this };

	auto &new_func = m_functions.emplace_back();
	// allocate the first block
	new_func.blocks.emplace_back();
	new_func.metadata.function_name = name;
	m_values.push_back(std::make_unique<BytecodeFunctionValue>(
		name, FunctionInfo{ m_functions.size() - 1, new_func, this }));
	static_cast<BytecodeFunctionValue *>(m_values.back().get())
		->function_info()
		.function.metadata.function_name = name;
	return static_cast<BytecodeFunctionValue *>(m_values.back().get());
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
	return std::make_shared<BytecodeProgram>(std::move(m_functions), filename, argv);
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

std::string BytecodeGenerator::mangle_namespace(std::stack<BytecodeGenerator::Scope> s) const
{
	std::string result = s.top().name;
	s.pop();
	while (!s.empty()) {
		result = s.top().name + '.' + std::move(result);
		s.pop();
	}
	return result;
}
}// namespace codegen