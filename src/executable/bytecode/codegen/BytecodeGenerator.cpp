#include "BytecodeGenerator.hpp"
#include "executable/bytecode/BytecodeProgram.hpp"
#include "executable/bytecode/instructions/BinaryOperation.hpp"
#include "executable/bytecode/instructions/BinarySubscript.hpp"
#include "executable/bytecode/instructions/BuildDict.hpp"
#include "executable/bytecode/instructions/BuildList.hpp"
#include "executable/bytecode/instructions/BuildTuple.hpp"
#include "executable/bytecode/instructions/ClearExceptionState.hpp"
#include "executable/bytecode/instructions/CompareOperation.hpp"
#include "executable/bytecode/instructions/DeleteName.hpp"
#include "executable/bytecode/instructions/DictMerge.hpp"
#include "executable/bytecode/instructions/ForIter.hpp"
#include "executable/bytecode/instructions/FunctionCall.hpp"
#include "executable/bytecode/instructions/FunctionCallEx.hpp"
#include "executable/bytecode/instructions/FunctionCallWithKeywords.hpp"
#include "executable/bytecode/instructions/GetIter.hpp"
#include "executable/bytecode/instructions/ImportName.hpp"
#include "executable/bytecode/instructions/InplaceAdd.hpp"
#include "executable/bytecode/instructions/InplaceSub.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"
#include "executable/bytecode/instructions/Jump.hpp"
#include "executable/bytecode/instructions/JumpForward.hpp"
#include "executable/bytecode/instructions/JumpIfFalse.hpp"
#include "executable/bytecode/instructions/JumpIfFalseOrPop.hpp"
#include "executable/bytecode/instructions/JumpIfNotExceptionMatch.hpp"
#include "executable/bytecode/instructions/JumpIfTrue.hpp"
#include "executable/bytecode/instructions/JumpIfTrueOrPop.hpp"
#include "executable/bytecode/instructions/ListExtend.hpp"
#include "executable/bytecode/instructions/ListToTuple.hpp"
#include "executable/bytecode/instructions/LoadAssertionError.hpp"
#include "executable/bytecode/instructions/LoadAttr.hpp"
#include "executable/bytecode/instructions/LoadBuildClass.hpp"
#include "executable/bytecode/instructions/LoadClosure.hpp"
#include "executable/bytecode/instructions/LoadConst.hpp"
#include "executable/bytecode/instructions/LoadDeref.hpp"
#include "executable/bytecode/instructions/LoadFast.hpp"
#include "executable/bytecode/instructions/LoadGlobal.hpp"
#include "executable/bytecode/instructions/LoadMethod.hpp"
#include "executable/bytecode/instructions/LoadName.hpp"
#include "executable/bytecode/instructions/MakeFunction.hpp"
#include "executable/bytecode/instructions/MethodCall.hpp"
#include "executable/bytecode/instructions/Move.hpp"
#include "executable/bytecode/instructions/RaiseVarargs.hpp"
#include "executable/bytecode/instructions/ReRaise.hpp"
#include "executable/bytecode/instructions/ReturnValue.hpp"
#include "executable/bytecode/instructions/SetupExceptionHandling.hpp"
#include "executable/bytecode/instructions/StoreAttr.hpp"
#include "executable/bytecode/instructions/StoreDeref.hpp"
#include "executable/bytecode/instructions/StoreFast.hpp"
#include "executable/bytecode/instructions/StoreGlobal.hpp"
#include "executable/bytecode/instructions/StoreName.hpp"
#include "executable/bytecode/instructions/StoreSubscript.hpp"
#include "executable/bytecode/instructions/Unary.hpp"
#include "executable/bytecode/instructions/UnpackSequence.hpp"
#include "executable/bytecode/instructions/WithExceptStart.hpp"

#include "ast/optimizers/ConstantFolding.hpp"
#include "executable/FunctionBlock.hpp"
#include "executable/Mangler.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"

#include "VariablesResolver.hpp"

#include <filesystem>

namespace fs = std::filesystem;

using namespace ast;

namespace codegen {

namespace {
	bool compare_values(const py::Value &lhs, const py::Value &rhs)
	{
		if (lhs.index() != rhs.index()) { return false; }
		return std::visit(
			overloaded{
				[&]<typename T>(const T &lhs_value) { return lhs_value == std::get<T>(rhs); },
				[&](const py::Number &lhs_value) {
					// make sure that doubles and integers are stored independently even when their
					// values are equivalent
					return lhs_value.value.index() == std::get<py::Number>(rhs).value.index()
						   && lhs_value == std::get<py::Number>(rhs);
				} },
			lhs);
	}
}// namespace


BytecodeValue *BytecodeGenerator::build_dict(const std::vector<Register> &key_registers,
	const std::vector<Register> &value_registers)
{
	auto *result = create_value();

	// FIXME: the move instructions below guarantee that the dictionary keys/values are contiguosly
	// layed out on the stack. Ideally this would be done immediately when generating the
	// keys/values
	if (!key_registers.empty()) {
		std::optional<size_t> offset;
		bool first = true;
		for (const auto &key : key_registers) {
			auto *dst = create_value();
			if (first) {
				offset = dst->get_register();
				first = false;
			}
			emit<Move>(dst->get_register(), key);
		}

		for (const auto &value : value_registers) {
			auto *dst = create_value();
			emit<Move>(dst->get_register(), value);
		}


		ASSERT(offset.has_value())
		ASSERT(key_registers.size() == value_registers.size())

		size_t size = key_registers.size();
		emit<BuildDict>(result->get_register(), size, *offset);
	} else {
		emit<BuildDict>(result->get_register(), 0, 0);
	}
	return result;
}

BytecodeValue *BytecodeGenerator::build_list(const std::vector<Register> &element_registers)
{
	auto *result = create_value();
	if (!element_registers.empty()) {
		std::optional<size_t> offset;
		bool first = true;
		for (const auto &el : element_registers) {
			auto *dst = create_value();
			if (first) {
				offset = dst->get_register();
				first = false;
			}
			emit<Move>(dst->get_register(), el);
		}
		ASSERT(offset.has_value())
		emit<BuildList>(result->get_register(), element_registers.size(), *offset);
	} else {
		emit<BuildList>(result->get_register(), 0, 0);
	}
	return result;
}

BytecodeValue *BytecodeGenerator::build_tuple(const std::vector<Register> &element_registers)
{
	auto *result = create_value();
	if (!element_registers.empty()) {
		std::optional<size_t> offset;
		bool first = true;
		for (const auto &el : element_registers) {
			auto *dst = create_value();
			if (first) {
				offset = dst->get_register();
				first = false;
			}
			emit<Move>(dst->get_register(), el);
		}
		ASSERT(offset.has_value())
		emit<BuildTuple>(result->get_register(), element_registers.size(), *offset);
	} else {
		emit<BuildTuple>(result->get_register(), 0, 0);
	}
	return result;
}

std::tuple<size_t, size_t> BytecodeGenerator::move_to_stack(const std::vector<Register> &args)
{
	if (args.empty()) { return { 0, 0 }; }
	std::optional<size_t> offset;
	bool first = true;
	for (const auto &arg : args) {
		auto *dst = create_value();
		if (first) {
			offset = dst->get_register();
			first = false;
		}
		emit<Move>(dst->get_register(), arg);
	}
	ASSERT(offset.has_value())
	return { args.size(), *offset };
}

void BytecodeGenerator::emit_call(Register func, const std::vector<Register> &args)
{
	if (!args.empty()) {
		const auto [args_size, stack_offset] = move_to_stack(args);
		emit<FunctionCall>(func, args_size, stack_offset);
	} else {
		emit<FunctionCall>(func, 0, 0);
	}
}

void BytecodeGenerator::make_function(Register dst,
	const std::string &name,
	const std::vector<Register> &defaults,
	const std::vector<Register> &kw_defaults,
	const std::optional<Register> &captures_tuple)
{
	auto *name_value_const = load_const(py::String{ name }, m_function_id);
	auto *name_value = create_value();
	emit<LoadConst>(name_value->get_register(), name_value_const->get_index());

	const auto [defaults_size, defaults_stack_offset] = move_to_stack(defaults);
	const auto [kw_defaults_size, kw_defaults_stack_offset] = move_to_stack(kw_defaults);

	emit<MakeFunction>(dst,
		name_value->get_register(),
		defaults_size,
		defaults_stack_offset,
		kw_defaults_size,
		kw_defaults_stack_offset,
		captures_tuple);
}

void BytecodeGenerator::store_name(const std::string &name, BytecodeValue *src)
{
	auto &varnames = std::next(m_functions.functions.begin(), m_function_id)->metadata.varnames;
	if (std::find(varnames.begin(), varnames.end(), name) != varnames.end()) {
		varnames.push_back(name);
	}
	const auto &scope_name = m_stack.top().mangled_name;
	const auto &visibility = [&] {
		if (auto it = m_variable_visibility.find(scope_name); it != m_variable_visibility.end()) {
			return it;
		} else {
			TODO();
		}
	}();

	const auto &name_visibility = [&] {
		if (auto it = visibility->second->visibility.find(name);
			it != visibility->second->visibility.end()) {
			return it->second;
		} else {
			TODO();
		}
	}();

	switch (name_visibility) {
	case VariablesResolver::Visibility::GLOBAL: {
		emit<StoreGlobal>(name, src->get_register());
	} break;
	case VariablesResolver::Visibility::NAME: {
		emit<StoreName>(name, src->get_register());
	} break;
	case VariablesResolver::Visibility::LOCAL: {
		auto *value = [&] {
			if (auto it = m_stack.top().locals.find(name); it != m_stack.top().locals.end()) {
				ASSERT(std::holds_alternative<BytecodeStackValue *>(it->second))
				return std::get<BytecodeStackValue *>(it->second);
			} else {
				auto *value = create_stack_value();
				m_stack.top().locals.emplace(name, value);
				return value;
			}
		}();
		emit<StoreFast>(value->get_stack_index(), name, src->get_register());
	} break;
	case VariablesResolver::Visibility::CELL:
	case VariablesResolver::Visibility::FREE: {
		auto *value = [&]() -> BytecodeFreeValue * {
			if (auto it = m_stack.top().locals.find(name); it != m_stack.top().locals.end()) {
				ASSERT(std::holds_alternative<BytecodeFreeValue *>(it->second))
				return std::get<BytecodeFreeValue *>(it->second);
			} else {
				auto *value = create_free_value();
				m_stack.top().locals.emplace(name, value);
				return value;
			}
		}();
		emit<StoreDeref>(value->get_free_var_index(), src->get_register());
	} break;
	}
}

BytecodeValue *BytecodeGenerator::load_name(const std::string &name)
{
	auto *dst = create_value();

	const auto &scope_name = m_stack.top().mangled_name;
	const auto &visibility = [&] {
		if (auto it = m_variable_visibility.find(scope_name); it != m_variable_visibility.end()) {
			return it;
		} else {
			TODO();
		}
	}();

	const auto &name_visibility = [&] {
		if (auto it = visibility->second->visibility.find(name);
			it != visibility->second->visibility.end()) {
			return it->second;
		} else {
			TODO();
		}
	}();

	switch (name_visibility) {
	case VariablesResolver::Visibility::GLOBAL: {
		emit<LoadGlobal>(dst->get_register(), name);
	} break;
	case VariablesResolver::Visibility::NAME: {
		emit<LoadName>(dst->get_register(), name);
	} break;
	case VariablesResolver::Visibility::LOCAL: {
		ASSERT(m_stack.top().locals.contains(name));
		const auto &l = m_stack.top().locals.at(name);
		ASSERT(std::holds_alternative<BytecodeStackValue *>(l))
		emit<LoadFast>(
			dst->get_register(), std::get<BytecodeStackValue *>(l)->get_stack_index(), name);
	} break;
	case VariablesResolver::Visibility::CELL:
	case VariablesResolver::Visibility::FREE: {
		ASSERT(m_stack.top().locals.contains(name));
		const auto &l = m_stack.top().locals.at(name);
		ASSERT(std::holds_alternative<BytecodeFreeValue *>(l))
		emit<LoadDeref>(
			dst->get_register(), std::get<BytecodeFreeValue *>(l)->get_free_var_index(), name);
	} break;
	}
	return dst;
}

BytecodeStaticValue *BytecodeGenerator::load_const(const py::Value &value, size_t function_id)
{
	auto &consts = std::next(m_functions.functions.begin(), function_id)->metadata.consts;
	for (size_t i = 0; const auto &static_value : consts) {
		if (compare_values(static_value, value)) {
			m_values.push_back(std::make_unique<BytecodeStaticValue>(i));
			return static_cast<BytecodeStaticValue *>(m_values.back().get());
		}
		++i;
	}
	consts.push_back(value);
	m_values.push_back(std::make_unique<BytecodeStaticValue>(consts.size() - 1));
	return static_cast<BytecodeStaticValue *>(m_values.back().get());
}

Value *BytecodeGenerator::visit(const Name *node)
{
	ASSERT(node->ids().size() == 1)
	return load_name(node->ids()[0]);
}


Value *BytecodeGenerator::visit(const Constant *node)
{
	auto *dst = create_value();
	auto *value = load_const(*node->value(), m_function_id);
	emit<LoadConst>(dst->get_register(), value->get_index());
	return dst;
}

Value *BytecodeGenerator::visit(const BinaryExpr *node)
{
	auto *lhs = generate(node->lhs().get(), m_function_id);
	auto *rhs = generate(node->rhs().get(), m_function_id);
	auto *dst = create_value();

	switch (node->op_type()) {
	case BinaryOpType::PLUS: {
		emit<BinaryOperation>(dst->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			BinaryOperation::Operation::PLUS);
	} break;
	case BinaryOpType::MINUS: {
		emit<BinaryOperation>(dst->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			BinaryOperation::Operation::MINUS);
	} break;
	case BinaryOpType::MULTIPLY: {
		emit<BinaryOperation>(dst->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			BinaryOperation::Operation::MULTIPLY);
	} break;
	case BinaryOpType::EXP: {
		emit<BinaryOperation>(dst->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			BinaryOperation::Operation::EXP);
	} break;
	case BinaryOpType::MODULO: {
		emit<BinaryOperation>(dst->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			BinaryOperation::Operation::MODULO);
	} break;
	case BinaryOpType::SLASH: {
		emit<BinaryOperation>(dst->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			BinaryOperation::Operation::SLASH);
	} break;
	case BinaryOpType::FLOORDIV:
		TODO();
	case BinaryOpType::LEFTSHIFT: {
		emit<BinaryOperation>(dst->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			BinaryOperation::Operation::LEFTSHIFT);
	} break;
	case BinaryOpType::RIGHTSHIFT:
		TODO();
	}

	return dst;
}

Value *BytecodeGenerator::visit(const FunctionDefinition *node)
{
	std::vector<BytecodeValue *> decorator_functions;
	decorator_functions.reserve(node->decorator_list().size());
	for (const auto &decorator_function : node->decorator_list()) {
		auto *f = generate(decorator_function.get(), m_function_id);
		ASSERT(f)
		decorator_functions.push_back(f);
	}

	std::vector<std::string> varnames;
	std::vector<size_t> cell2arg;

	m_ctx.push_local_args(node->args());
	const std::string &function_name = Mangler::default_mangler().function_mangle(
		mangle_namespace(m_stack), node->name(), node->source_location());
	auto *f = create_function(function_name);

	create_nested_scope(node->name(), function_name);
	std::vector<std::pair<std::string, BytecodeFreeValue *>> captures;
	for (const auto &capture : m_variable_visibility.at(function_name)->captures) {
		auto *value = create_free_value();
		captures.emplace_back(capture, value);
		m_stack.top().locals.emplace(capture, value);
	}

	auto *block = allocate_block(f->function_info().function_id);
	auto *old_block = m_current_block;
	set_insert_point(block);

	generate(node->args().get(), f->function_info().function_id);

	for (const auto &node : node->body()) { generate(node.get(), f->function_info().function_id); }

	// always return None
	// this can be optimised away later on
	auto none_value_register = allocate_register();
	auto *value = load_const(py::NameConstant{ py::NoneType{} }, f->function_info().function_id);
	emit<LoadConst>(none_value_register, value->get_index());
	emit<ReturnValue>(none_value_register);

	const auto &name_visibility_it = m_variable_visibility.find(function_name);
	ASSERT(name_visibility_it != m_variable_visibility.end())
	const auto &name_visibility = name_visibility_it->second->visibility;

	for (size_t idx = 0; const auto &arg_name : node->args()->argument_names()) {
		varnames.push_back(arg_name);
		ASSERT(name_visibility.find(arg_name) != name_visibility.end())
		if (auto it = name_visibility.find(arg_name);
			it->second == VariablesResolver::Visibility::CELL) {
			cell2arg.push_back(idx);
		}
		idx++;
	}
	for (size_t idx = node->args()->argument_names().size();
		 const auto &arg_name : node->args()->kw_only_argument_names()) {
		varnames.push_back(arg_name);
		ASSERT(name_visibility.find(arg_name) != name_visibility.end())
		if (auto it = name_visibility.find(arg_name);
			it->second == VariablesResolver::Visibility::CELL) {
			cell2arg.push_back(idx);
		}
		idx++;
	}

	set_insert_point(old_block);
	m_ctx.pop_local_args();
	m_stack.pop();
	exit_function(f->function_info().function_id);

	size_t arg_count = node->args()->args().size();
	size_t kwonly_arg_count = node->args()->kwonlyargs().size();

	std::vector<Register> defaults;
	defaults.reserve(node->args()->defaults().size());
	for (const auto &default_node : node->args()->defaults()) {
		defaults.push_back(generate(default_node.get(), m_function_id)->get_register());
	}

	std::vector<Register> kw_defaults;
	kw_defaults.reserve(node->args()->kw_defaults().size());
	for (const auto &default_node : node->args()->kw_defaults()) {
		if (default_node) {
			kw_defaults.push_back(generate(default_node.get(), m_function_id)->get_register());
		}
	}

	auto captures_tuple = [&]() -> std::optional<Register> {
		if (!captures.empty()) {
			std::vector<Register> capture_regs;
			capture_regs.reserve(captures.size());
			for (const auto &[name, el] : captures) {
				ASSERT(m_stack.top().locals.contains(name));
				const auto &value = m_stack.top().locals.at(name);
				ASSERT(std::holds_alternative<BytecodeFreeValue *>(value))
				emit<LoadClosure>(el->get_free_var_index(),
					std::get<BytecodeFreeValue *>(value)->get_free_var_index(),
					name);
				capture_regs.push_back(el->get_free_var_index());
			}
			auto *tuple_value = build_tuple(capture_regs);
			return tuple_value->get_register();
		} else {
			return {};
		}
	}();

	auto flags = CodeFlags::create();
	if (node->args()->vararg() != nullptr) { flags.set(CodeFlags::Flag::VARARGS); }
	if (node->args()->kwarg() != nullptr) { flags.set(CodeFlags::Flag::VARKEYWORDS); }

	f->function_info().function.metadata.varnames = varnames;

	for (const auto &[varname, v] : name_visibility) {
		if (v == VariablesResolver::Visibility::FREE) {
			f->function_info().function.metadata.freevars.push_back(varname);
		} else if (v == VariablesResolver::Visibility::CELL) {
			f->function_info().function.metadata.cellvars.push_back(varname);
		} else if (v == VariablesResolver::Visibility::LOCAL) {
			f->function_info().function.metadata.varnames.push_back(varname);
		} else {
			// TODO: add to co_names
			// A tuple containing names used by the bytecode:
			//  * global variables,
			//  * functions
			//  * classes
			//	* attributes loaded from objects
		}
	}

	// TODO
	// f->function_info().function.metadata.filename = ;
	f->function_info().function.metadata.arg_count = arg_count;
	f->function_info().function.metadata.kwonly_arg_count = kwonly_arg_count;
	f->function_info().function.metadata.cell2arg = std::move(cell2arg);
	f->function_info().function.metadata.nlocals = varnames.size();
	f->function_info().function.metadata.flags = flags;

	make_function(f->get_register(), f->get_name(), defaults, kw_defaults, captures_tuple);

	store_name(node->name(), f);
	if (!decorator_functions.empty()) {
		std::vector<BytecodeValue *> args;
		auto *function = load_name(node->name());
		args.push_back(function);
		for (int32_t i = decorator_functions.size() - 1; i >= 0; --i) {
			const auto &decorator_function = decorator_functions[i];
			emit_call(decorator_function->get_register(),
				std::vector<Register>{ args.back()->get_register() });
			args.clear();
			args.push_back(create_return_value());
		}
		store_name(node->name(), args.back());
	}
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

Value *BytecodeGenerator::visit(const Argument *node)
{
	const auto &var_scope =
		m_variable_visibility.at(m_stack.top().mangled_name)->visibility.at(node->name());
	switch (var_scope) {
	case VariablesResolver::Visibility::CELL: {
		m_stack.top().locals.emplace(node->name(), create_free_value());
		auto f = std::next(m_functions.functions.begin(), m_function_id);
		f->metadata.varnames.push_back(node->name());
	} break;
	case VariablesResolver::Visibility::FREE: {
		TODO();
	} break;
	case VariablesResolver::Visibility::LOCAL: {
		m_stack.top().locals.emplace(node->name(), create_stack_value());
	} break;
	case VariablesResolver::Visibility::GLOBAL: {
		TODO();
	} break;
	case VariablesResolver::Visibility::NAME:
		m_stack.top().locals.emplace(node->name(), create_value());
	}
	return nullptr;
}

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
			for (const auto &var : ast_name->ids()) { store_name(var, src); }
		} else if (auto ast_attr = as<Attribute>(target)) {
			auto *dst = generate(ast_attr->value().get(), m_function_id);
			emit<StoreAttr>(dst->get_register(), src->get_register(), ast_attr->attr());
		} else if (auto ast_tuple = as<Tuple>(target)) {
			std::vector<BytecodeValue *> dst_values;
			std::vector<Register> dst_registers;
			for (size_t i = 0; i < ast_tuple->elements().size(); ++i) {
				dst_values.push_back(create_value());
				dst_registers.push_back(dst_values.back()->get_register());
			}
			emit<UnpackSequence>(dst_registers, src->get_register());
			for (size_t idx = 0; const auto &dst_value : dst_values) {
				const auto &el = ast_tuple->elements()[idx++];
				store_name(as<Name>(el)->ids()[0], dst_value);
			}
		} else if (auto ast_subscript = as<Subscript>(target)) {
			auto *obj = generate(ast_subscript->value().get(), m_function_id);
			const auto &slice = ast_subscript->slice();
			auto *index = [&]() -> BytecodeValue * {
				if (std::holds_alternative<Subscript::Index>(slice)) {
					return generate(std::get<Subscript::Index>(slice).value.get(), m_function_id);
				} else {
					TODO();
				}
			}();
			emit<StoreSubscript>(obj->get_register(), index->get_register(), src->get_register());
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
		BytecodeValue *list_value = nullptr;
		bool first_args_expansion = true;
		std::vector<Register> args_lhs;
		for (const auto &arg : node->args()) {
			if (is_args_expansion(arg)) {
				if (first_args_expansion) {
					list_value = build_list(args_lhs);
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
		if (first_args_expansion) { list_value = build_list(args_lhs); }
		auto *args_tuple = create_value();
		emit<ListToTuple>(args_tuple->get_register(), list_value->get_register());
		arg_values.push_back(args_tuple);

		if (requires_kwargs_expansion) {
			BytecodeValue *dict_value = nullptr;
			std::vector<Register> key_registers;
			std::vector<Register> value_registers;
			bool first_kwargs_expansion = true;

			for (const auto &el : node->keywords()) {
				if (is_kwargs_expansion(el)) {
					if (first_kwargs_expansion) {
						ASSERT(key_registers.size() == value_registers.size())
						dict_value = build_dict(key_registers, value_registers);
						key_registers.clear();
						value_registers.clear();
						first_kwargs_expansion = false;
					}
					auto *kwargs_dict = generate(el->value().get(), m_function_id);
					emit<DictMerge>(dict_value->get_register(), kwargs_dict->get_register());
				} else {
					const auto &name = *el->arg();
					auto *key = create_value();
					auto *value = generate(el.get(), m_function_id);
					emit<LoadConst>(key->get_register(),
						load_const(py::String{ name }, m_function_id)->get_index());
					if (first_kwargs_expansion) {
						key_registers.push_back(key->get_register());
						value_registers.push_back(value->get_register());
					} else {
						auto *new_dict = build_dict(key_registers, value_registers);
						emit<DictMerge>(dict_value->get_register(), new_dict->get_register());
					}
				}
			}
			ASSERT(first_kwargs_expansion == false)
			ASSERT(dict_value)
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
				emit_call(func->get_register(), std::move(arg_registers));
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
	auto *iter_variable = create_value();

	// call the __iter__ implementation
	emit<GetIter>(iterator_register, iterator_func->get_register());

	bind(*forloop_start_label);
	emit<ForIter>(iter_variable->get_register(), iterator_register, forloop_end_label);

	// call the __next__ implementation
	if (auto target = as<Name>(node->target())) {
		auto target_ids = target->ids();
		if (target_ids.size() != 1) { TODO(); }
		auto target_name = target_ids[0];
		store_name(target_name, iter_variable);
	} else if (auto target = as<Tuple>(node->target())) {
		for (const auto &el : target->elements()) {
			auto name = as<Name>(el);
			ASSERT(name);
			auto target_ids = name->ids();
			if (target_ids.size() != 1) { TODO(); }
			auto target_name = target_ids[0];
			store_name(target_name, iter_variable);
		}
	}

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
		emit<CompareOperation>(result->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			CompareOperation::Comparisson::Eq);
	} break;
	case Compare::OpType::NotEq: {
		emit<CompareOperation>(result->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			CompareOperation::Comparisson::NotEq);
	} break;
	case Compare::OpType::Lt: {
		emit<CompareOperation>(result->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			CompareOperation::Comparisson::Lt);
	} break;
	case Compare::OpType::LtE: {
		emit<CompareOperation>(result->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			CompareOperation::Comparisson::LtE);
	} break;
	case Compare::OpType::Gt: {
		emit<CompareOperation>(result->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			CompareOperation::Comparisson::Gt);
	} break;
	case Compare::OpType::GtE: {
		emit<CompareOperation>(result->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			CompareOperation::Comparisson::GtE);
	} break;
	case Compare::OpType::Is: {
		emit<CompareOperation>(result->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			CompareOperation::Comparisson::Is);
	} break;
	case Compare::OpType::IsNot: {
		emit<CompareOperation>(result->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			CompareOperation::Comparisson::IsNot);
	} break;
	case Compare::OpType::In: {
		emit<CompareOperation>(result->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			CompareOperation::Comparisson::In);
	} break;
	case Compare::OpType::NotIn: {
		emit<CompareOperation>(result->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			CompareOperation::Comparisson::NotIn);
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

	return build_list(element_registers);
}

Value *BytecodeGenerator::visit(const Tuple *node)
{
	std::vector<Register> element_registers;
	element_registers.reserve(node->elements().size());

	for (const auto &el : node->elements()) {
		auto *element_value = generate(el.get(), m_function_id);
		element_registers.push_back(element_value->get_register());
	}

	return build_tuple(element_registers);
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
		create_nested_scope(node->name(), class_mangled_name);
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
		emit<LoadConst>(
			qualname_register, load_const(py::String{ node->name() }, class_id)->get_index());
		emit<StoreName>("__qualname__", qualname_register);

		// the actual class definition
		for (const auto &el : node->body()) { generate(el.get(), class_id); }

		emit<LoadConst>(return_none_register,
			load_const(py::NameConstant{ py::NoneType{} }, class_id)->get_index());
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
	emit<LoadConst>(class_name_register,
		load_const(py::String{ class_mangled_name }, m_function_id)->get_index());
	emit<LoadConst>(class_location_register,
		load_const(py::Number{ static_cast<int64_t>(class_id) }, m_function_id)->get_index());

	if (kwarg_registers.empty()) {
		emit_call(builtin_build_class_register, std::move(arg_registers));
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

	return build_dict(key_registers, value_registers);
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
		store_name(named_target->ids()[0], lhs);
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
	auto *module_value = create_value();
	emit<ImportName>(module_value->get_register(), node->names());
	if (node->asname().has_value()) {
		store_name(*node->asname(), module_value);
	} else {
		store_name(node->names()[0], module_value);
	}
	return nullptr;
}

Value *BytecodeGenerator::visit(const Module *node)
{
	const auto &module_name = fs::path(node->filename()).stem();
	create_nested_scope(module_name, module_name);
	BytecodeValue *last = nullptr;
	for (const auto &statement : node->body()) { last = generate(statement.get(), m_function_id); }

	// TODO: should the module return the last value if there is one?
	last = create_value();
	emit<LoadConst>(last->get_register(),
		load_const(py::NameConstant{ py::NoneType{} }, m_function_id)->get_index());
	emit<ReturnValue>(last->get_register());
	m_stack.pop();
	return last;
}

Value *BytecodeGenerator::visit(const Subscript *node)
{
	auto *result = create_value();
	const auto *value = generate(node->value().get(), m_function_id);
	if (std::holds_alternative<Subscript::Index>(node->slice())) {
		const auto *index =
			generate(std::get<Subscript::Index>(node->slice()).value.get(), m_function_id);
		emit<BinarySubscript>(result->get_register(), value->get_register(), index->get_register());
		return result;
	} else if (std::holds_alternative<Subscript::Slice>(node->slice())) {
		TODO();
	} else if (std::holds_alternative<Subscript::ExtSlice>(node->slice())) {
		TODO();
	} else {
		TODO();
	}
	return nullptr;
}

Value *BytecodeGenerator::visit(const Raise *node)
{
	if (node->cause()) {
		ASSERT(node->exception())
		const auto *exception = generate(node->exception().get(), m_function_id);
		const auto *cause = generate(node->cause().get(), m_function_id);
		emit<RaiseVarargs>(exception->get_register(), cause->get_register());
	} else if (node->exception()) {
		const auto *exception = generate(node->exception().get(), m_function_id);
		emit<RaiseVarargs>(exception->get_register());
	} else {
		emit<RaiseVarargs>();
	}
	return nullptr;
}

Value *BytecodeGenerator::visit(const With *node)
{
	static size_t exit_label_count = 0;
	auto exit_label = make_label(fmt::format("BOOL_OP_END_{}", exit_label_count++), m_function_id);

	auto *body_block = allocate_block(m_function_id);
	auto *cleanup_block = allocate_block(m_function_id);

	if (node->items().size() > 1) { TODO(); }
	std::vector<BytecodeValue *> with_item_results;

	for (const auto &item : node->items()) {
		with_item_results.push_back(
			static_cast<BytecodeValue *>(generate(item.get(), m_function_id)));
	}

	emit<SetupExceptionHandling>();

	set_insert_point(body_block);
	for (const auto &statement : node->body()) { generate(statement.get(), m_function_id); }

	set_insert_point(cleanup_block);
	for (const auto &item : with_item_results) {
		auto *exit_result = create_value();
		auto *exit_method = create_value();

		emit<LoadMethod>(exit_method->get_register(), item->get_register(), "__exit__");
		emit<WithExceptStart>(exit_result->get_register(), exit_method->get_register());
		emit<JumpIfTrue>(exit_result->get_register(), exit_label);
	}
	emit<ReRaise>();

	bind(*exit_label);
	emit<ClearExceptionState>();

	return nullptr;
}

Value *BytecodeGenerator::visit(const WithItem *node)
{
	auto *ctx_expr_result = generate(node->context_expr().get(), m_function_id);
	auto *enter_method = create_value();
	auto *ctx_expr = create_value();
	emit<Move>(ctx_expr->get_register(), ctx_expr_result->get_register());
	emit<LoadMethod>(enter_method->get_register(), ctx_expr->get_register(), "__enter__");
	emit<MethodCall>(enter_method->get_register(), "__enter__", std::vector<Register>{});
	auto *enter_result = create_return_value();

	if (auto optional_vars = node->optional_vars()) {
		if (auto name = as<Name>(optional_vars)) {
			ASSERT(as<Name>(optional_vars)->ids().size() == 1)
			store_name(as<Name>(optional_vars)->ids()[0], enter_result);
		} else if (auto tuple = as<Tuple>(optional_vars)) {
			(void)tuple;
			TODO();
		} else if (auto list = as<List>(optional_vars)) {
			(void)list;
			TODO();
		} else {
			ASSERT_NOT_REACHED();
		}
	}

	return ctx_expr;
}

Value *BytecodeGenerator::visit(const IfExpr *node)
{
	static size_t if_expr_count = 0;

	auto orelse_start_label =
		make_label(fmt::format("IF_EXPR_ORELSE_{}", if_expr_count), m_function_id);
	auto end_label = make_label(fmt::format("IF_EXPR_END_{}", if_expr_count++), m_function_id);

	auto return_value = create_value();
	// if
	auto *test_result = generate(node->test().get(), m_function_id);
	emit<JumpIfFalse>(test_result->get_register(), orelse_start_label);
	auto *if_result = generate(node->body().get(), m_function_id);
	ASSERT(if_result)
	emit<Move>(return_value->get_register(), if_result->get_register());
	emit<Jump>(end_label);

	// else
	bind(*orelse_start_label);
	auto *else_result = generate(node->orelse().get(), m_function_id);
	emit<Move>(return_value->get_register(), else_result->get_register());
	bind(*end_label);

	return return_value;
}

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

	auto *finally_block_with_reraise = allocate_block(m_function_id);
	auto *finally_block = allocate_block(m_function_id);

	emit<SetupExceptionHandling>();
	set_insert_point(body_block);

	for (const auto &statement : node->body()) { generate(statement.get(), m_function_id); }

	emit<JumpForward>(list_node_distance(function(m_function_id), body_block, finally_block));

	for (size_t idx = 0; const auto &handler : node->handlers()) {
		auto *exception_handler_block = exception_handler_blocks[idx];
		set_insert_point(exception_handler_block);
		if (!handler->type()) {
			if ((idx / 2) != node->handlers().size() - 1) {
				// FIXME: implement SyntaxError and error throwing when parsing source code
				spdlog::error("SyntaxError: default 'except:' must be last");
				std::abort();
			}
		} else {
			auto *exception_type = generate(handler->type().get(), m_function_id);
			emit<JumpIfNotExceptionMatch>(exception_type->get_register());
		}
		idx++;
		set_insert_point(exception_handler_blocks[idx]);
		for (const auto &el : handler->body()) { generate(el.get(), m_function_id); }
		emit<ClearExceptionState>();
		emit<JumpForward>(list_node_distance(
			function(m_function_id), exception_handler_blocks[idx], finally_block));
		idx++;
	}

	set_insert_point(finally_block_with_reraise);
	for (const auto &statement : node->finalbody()) { generate(statement.get(), m_function_id); }
	emit<ReRaise>();
	emit<JumpForward>(2);

	set_insert_point(finally_block);
	for (const auto &statement : node->finalbody()) { generate(statement.get(), m_function_id); }

	auto *next_block = allocate_block(m_function_id);
	set_insert_point(next_block);

	return nullptr;
}

Value *BytecodeGenerator::visit(const ExceptHandler *) { TODO(); }

Value *BytecodeGenerator::visit(const Expression *node)
{
	return generate(node->value().get(), m_function_id);
}

Value *BytecodeGenerator::visit(const Global *) { return nullptr; }

Value *BytecodeGenerator::visit(const Delete *node)
{
	for (const auto &target : node->targets()) {
		const auto *value = generate(target.get(), m_function_id);
		emit<DeleteName>(value->get_register());
	}
	return nullptr;
}

Value *BytecodeGenerator::visit(const UnaryExpr *node)
{
	const auto *src = generate(node->operand().get(), m_function_id);
	auto *dst = create_value();
	switch (node->op_type()) {
	case UnaryOpType::ADD: {
		emit<Unary>(dst->get_register(), src->get_register(), Unary::Operation::POSITIVE);
	} break;
	case UnaryOpType::SUB: {
		emit<Unary>(dst->get_register(), src->get_register(), Unary::Operation::NEGATIVE);
	} break;
	case UnaryOpType::INVERT: {
		emit<Unary>(dst->get_register(), src->get_register(), Unary::Operation::INVERT);
	} break;
	case UnaryOpType::NOT: {
		emit<Unary>(dst->get_register(), src->get_register(), Unary::Operation::NOT);
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

	emit_call(assertion_function->get_register(), std::move(args));
	auto *exception = create_return_value();
	emit<RaiseVarargs>(exception->get_register());
	bind(*end_label);

	return nullptr;
}

Value *BytecodeGenerator::visit(const Pass *) { return nullptr; }

Value *BytecodeGenerator::visit(const NamedExpr *node)
{
	ASSERT(as<Name>(node->target()))
	ASSERT(as<Name>(node->target())->context_type() == ContextType::STORE)
	ASSERT(as<Name>(node->target())->ids().size() == 1)

	auto *dst = create_value();
	auto *src = generate(node->value().get(), m_function_id);
	emit<Move>(dst->get_register(), src->get_register());
	store_name(as<Name>(node->target())->ids()[0], src);

	return dst;
}

Value *BytecodeGenerator::visit(const JoinedStr *) { TODO(); }

Value *BytecodeGenerator::visit(const FormattedValue *) { TODO(); }

FunctionInfo::FunctionInfo(size_t function_id_, FunctionBlock &f, BytecodeGenerator *generator_)
	: function_id(function_id_), function(f), generator(generator_)
{
	generator->enter_function();
}

BytecodeGenerator::BytecodeGenerator()
{
	m_frame_register_count.push_back(0u);
	m_frame_stack_value_count.push_back(0u);
	m_frame_free_var_count.push_back(0u);
	(void)create_function("__main__entry__");
	m_current_block = &m_functions.functions.back().blocks.back();
}

BytecodeGenerator::~BytecodeGenerator() {}

void BytecodeGenerator::exit_function(size_t function_id)
{
	ASSERT(function_id < m_functions.functions.size())
	auto function = std::next(m_functions.functions.begin(), function_id);
	function->metadata.register_count = register_count();
	function->metadata.stack_size = stack_variable_count() + free_variable_count();
	m_frame_register_count.pop_back();
	m_frame_stack_value_count.pop_back();
	m_frame_free_var_count.pop_back();
}

BytecodeFunctionValue *BytecodeGenerator::create_function(const std::string &name)
{
	auto &new_func = m_functions.functions.emplace_back();
	m_function_map.emplace(name, std::ref(new_func));

	// allocate the first block
	new_func.blocks.emplace_back();
	new_func.metadata.function_name = name;
	m_values.push_back(std::make_unique<BytecodeFunctionValue>(name,
		allocate_register(),
		FunctionInfo{ m_functions.functions.size() - 1, new_func, this }));
	static_cast<BytecodeFunctionValue *>(m_values.back().get())
		->function_info()
		.function.metadata.function_name = name;
	return static_cast<BytecodeFunctionValue *>(m_values.back().get());
}

void BytecodeGenerator::relocate_labels(const FunctionBlocks &functions)
{
	for (const auto &function : functions.functions) {
		size_t instruction_idx{ 0 };
		for (const auto &block : function.blocks) {
			for (const auto &ins : block) { ins->relocate(*this, instruction_idx++); }
		}
	}
}

std::unique_ptr<Program> BytecodeGenerator::generate_executable(std::string filename,
	std::vector<std::string> argv)
{
	ASSERT(m_frame_register_count.size() == 2)
	ASSERT(m_frame_stack_value_count.size() == 2)
	ASSERT(m_frame_free_var_count.size() == 2)
	relocate_labels(m_functions);
	return std::make_unique<BytecodeProgram>(std::move(m_functions), filename, argv);
}

InstructionBlock *BytecodeGenerator::allocate_block(size_t function_id)
{
	ASSERT(function_id < m_functions.functions.size())

	auto function = std::next(m_functions.functions.begin(), function_id);
	auto &new_block = function->blocks.emplace_back();
	return &new_block;
}

std::unique_ptr<Program> BytecodeGenerator::compile(std::shared_ptr<ast::ASTNode> node,
	std::vector<std::string> argv,
	compiler::OptimizationLevel lvl)
{
	auto module = as<ast::Module>(node);
	ASSERT(module)

	if (lvl > compiler::OptimizationLevel::None) { ast::optimizer::constant_folding(node); }

	auto generator = BytecodeGenerator();

	generator.m_variable_visibility = VariablesResolver::resolve(module.get());

	for (const auto &[scope_name, scope] : generator.m_variable_visibility) {
		spdlog::debug("Scope name: {}", scope_name);
		for (const auto &[k, v] : scope->visibility) {
			if (v == VariablesResolver::Visibility::NAME) {
				spdlog::debug("  - {}: NAME", k);
			} else if (v == VariablesResolver::Visibility::LOCAL) {
				spdlog::debug("  - {}: LOCAL", k);
			} else if (v == VariablesResolver::Visibility::FREE) {
				spdlog::debug("  - {}: FREE", k);
			} else if (v == VariablesResolver::Visibility::CELL) {
				spdlog::debug("  - {}: CELL", k);
			} else if (v == VariablesResolver::Visibility::GLOBAL) {
				spdlog::debug("  - {}: GLOBAL", k);
			}
		}
	}

	node->codegen(&generator);

	// allocate registers for __main__
	generator.m_functions.functions.front().metadata.register_count = generator.register_count();
	generator.m_functions.functions.front().metadata.stack_size = 0;

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

void BytecodeGenerator::create_nested_scope(const std::string &name,
	const std::string &mangled_name)
{
	decltype(Scope::locals) locals;
	if (!m_stack.empty()) {
		for (const auto &[k, v] : m_stack.top().locals) {
			if (std::holds_alternative<BytecodeFreeValue *>(v)) { locals[k] = v; }
		}
	}
	m_stack.push(Scope{ .name = name, .mangled_name = mangled_name, .locals = locals });
}
}// namespace codegen