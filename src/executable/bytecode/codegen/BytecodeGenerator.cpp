#include "BytecodeGenerator.hpp"
#include "executable/bytecode/BytecodeProgram.hpp"
#include "executable/bytecode/instructions/BinaryOperation.hpp"
#include "executable/bytecode/instructions/BinarySubscript.hpp"
#include "executable/bytecode/instructions/BuildDict.hpp"
#include "executable/bytecode/instructions/BuildList.hpp"
#include "executable/bytecode/instructions/BuildSet.hpp"
#include "executable/bytecode/instructions/BuildSlice.hpp"
#include "executable/bytecode/instructions/BuildTuple.hpp"
#include "executable/bytecode/instructions/ClearExceptionState.hpp"
#include "executable/bytecode/instructions/ClearTopCleanup.hpp"
#include "executable/bytecode/instructions/CompareOperation.hpp"
#include "executable/bytecode/instructions/DeleteName.hpp"
#include "executable/bytecode/instructions/DeleteSubscript.hpp"
#include "executable/bytecode/instructions/DictAdd.hpp"
#include "executable/bytecode/instructions/DictMerge.hpp"
#include "executable/bytecode/instructions/DictUpdate.hpp"
#include "executable/bytecode/instructions/ForIter.hpp"
#include "executable/bytecode/instructions/FunctionCall.hpp"
#include "executable/bytecode/instructions/FunctionCallEx.hpp"
#include "executable/bytecode/instructions/FunctionCallWithKeywords.hpp"
#include "executable/bytecode/instructions/GetIter.hpp"
#include "executable/bytecode/instructions/GetYieldFromIter.hpp"
#include "executable/bytecode/instructions/ImportFrom.hpp"
#include "executable/bytecode/instructions/ImportName.hpp"
#include "executable/bytecode/instructions/ImportStar.hpp"
#include "executable/bytecode/instructions/InplaceOp.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"
#include "executable/bytecode/instructions/Jump.hpp"
#include "executable/bytecode/instructions/JumpForward.hpp"
#include "executable/bytecode/instructions/JumpIfFalse.hpp"
#include "executable/bytecode/instructions/JumpIfFalseOrPop.hpp"
#include "executable/bytecode/instructions/JumpIfNotExceptionMatch.hpp"
#include "executable/bytecode/instructions/JumpIfTrue.hpp"
#include "executable/bytecode/instructions/JumpIfTrueOrPop.hpp"
#include "executable/bytecode/instructions/LeaveExceptionHandling.hpp"
#include "executable/bytecode/instructions/ListAppend.hpp"
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
#include "executable/bytecode/instructions/SetAdd.hpp"
#include "executable/bytecode/instructions/SetupExceptionHandling.hpp"
#include "executable/bytecode/instructions/SetupWith.hpp"
#include "executable/bytecode/instructions/StoreAttr.hpp"
#include "executable/bytecode/instructions/StoreDeref.hpp"
#include "executable/bytecode/instructions/StoreFast.hpp"
#include "executable/bytecode/instructions/StoreGlobal.hpp"
#include "executable/bytecode/instructions/StoreName.hpp"
#include "executable/bytecode/instructions/StoreSubscript.hpp"
#include "executable/bytecode/instructions/Unary.hpp"
#include "executable/bytecode/instructions/UnpackSequence.hpp"
#include "executable/bytecode/instructions/WithExceptStart.hpp"
#include "executable/bytecode/instructions/YieldFrom.hpp"
#include "executable/bytecode/instructions/YieldLoad.hpp"
#include "executable/bytecode/instructions/YieldValue.hpp"

#include "ast/optimizers/ConstantFolding.hpp"
#include "executable/FunctionBlock.hpp"
#include "executable/Mangler.hpp"
#include "executable/Program.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"
#include "runtime/Value.hpp"

#include "VariablesResolver.hpp"

#include <filesystem>

namespace fs = std::filesystem;

using namespace ast;

namespace codegen {

namespace {
	class HasReturnVisitor : public NodeVisitor
	{
		bool m_has_return{ false };

		void visit(Return *) override { m_has_return = true; }
		void visit(FunctionDefinition *) override {}
		void visit(ClassDefinition *) override {}

	  public:
		static bool evaluate(ASTNode *node)
		{
			HasReturnVisitor visitor;
			visitor.dispatch(node);
			return visitor.m_has_return;
		}

		static bool evaluate(const std::vector<std::shared_ptr<ASTNode>> &nodes)
		{
			return std::any_of(
				nodes.begin(), nodes.end(), [](const auto &node) { return evaluate(node.get()); });
		}
	};

	class InlineFinallyReturnVisitor : public NodeTransformVisitor
	{
		const std::vector<std::shared_ptr<ASTNode>> &m_finally_nodes;

		std::vector<std::shared_ptr<ASTNode>> visit(std::shared_ptr<Return> node) override
		{
			ASSERT(m_can_return_multiple_nodes)
			auto new_nodes = m_finally_nodes;
			new_nodes.push_back(std::make_shared<Return>(*node));
			return new_nodes;
		}

		std::vector<std::shared_ptr<ASTNode>> visit(std::shared_ptr<FunctionDefinition>) override
		{
			return {};
		}
		std::vector<std::shared_ptr<ASTNode>> visit(std::shared_ptr<ClassDefinition>) override
		{
			return {};
		}

		InlineFinallyReturnVisitor(const std::vector<std::shared_ptr<ASTNode>> &finally_nodes)
			: m_finally_nodes(finally_nodes)
		{}

	  public:
		static void evaluate(std::vector<std::shared_ptr<ASTNode>> &block,
			const std::vector<std::shared_ptr<ASTNode>> &finally_nodes)
		{
			InlineFinallyReturnVisitor visitor{ finally_nodes };
			visitor.transform_multiple_nodes(block);
		}
	};

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

BytecodeValue *BytecodeGenerator::build_dict_simple(
	const std::vector<std::optional<Register>> &key_registers,
	const std::vector<Register> &value_registers)
{
	ASSERT(std::all_of(
		key_registers.begin(), key_registers.end(), [](const auto &el) { return el.has_value(); }));

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
			emit<Move>(dst->get_register(), *key);
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

BytecodeValue *BytecodeGenerator::build_dict(
	const std::vector<std::optional<Register>> &key_registers,
	const std::vector<Register> &value_registers)
{
	if (std::any_of(key_registers.begin(), key_registers.end(), [](const auto &el) {
			return !el.has_value();
		})) {
		ASSERT(key_registers.size() == value_registers.size());
		size_t begin_key_index = 0;
		size_t last_key_index = begin_key_index;
		BytecodeValue *dict = nullptr;
		while (last_key_index != key_registers.size()) {
			begin_key_index = last_key_index;
			for (; last_key_index < key_registers.size(); ++last_key_index) {
				if (!key_registers[last_key_index].has_value()) { break; }
			}
			std::vector<std::optional<Register>> simple_key_registers{
				key_registers.begin() + begin_key_index, key_registers.begin() + last_key_index
			};
			std::vector<Register> simple_value_registers{ value_registers.begin() + begin_key_index,
				value_registers.begin() + last_key_index };
			auto *tmp_dict = build_dict_simple(simple_key_registers, simple_value_registers);
			if (!dict) {
				dict = tmp_dict;
			} else {
				emit<DictUpdate>(dict->get_register(), tmp_dict->get_register());
			}
			const auto &value = value_registers[last_key_index++];
			emit<DictUpdate>(dict->get_register(), value);
		}
		ASSERT(dict);
		return dict;
	} else {
		return build_dict_simple(key_registers, value_registers);
	}
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

BytecodeValue *BytecodeGenerator::build_set(const std::vector<Register> &element_registers)
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
		emit<BuildSet>(result->get_register(), element_registers.size(), *offset);
	} else {
		emit<BuildSet>(result->get_register(), 0, 0);
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
		emit<StoreGlobal>(load_name(name, m_function_id)->get_index(), src->get_register());
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
		emit<StoreFast>(value->get_stack_index(), src->get_register());
	} break;
	case VariablesResolver::Visibility::CELL:
	case VariablesResolver::Visibility::FREE: {
		auto *value = [&]() -> BytecodeFreeValue * {
			if (auto it = m_stack.top().locals.find(name); it != m_stack.top().locals.end()) {
				ASSERT(std::holds_alternative<BytecodeFreeValue *>(it->second))
				return std::get<BytecodeFreeValue *>(it->second);
			} else {
				auto *value = create_free_value(name);
				m_stack.top().locals.emplace(name, value);
				return value;
			}
		}();
		emit<StoreDeref>(value->get_free_var_index(), src->get_register());
	} break;
	}
}

BytecodeValue *BytecodeGenerator::load_var(const std::string &name)
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
		emit<LoadGlobal>(dst->get_register(), load_name(name, m_function_id)->get_index());
	} break;
	case VariablesResolver::Visibility::NAME: {
		emit<LoadName>(dst->get_register(), name);
	} break;
	case VariablesResolver::Visibility::LOCAL: {
		ASSERT(m_stack.top().locals.contains(name));
		const auto &l = m_stack.top().locals.at(name);
		ASSERT(std::holds_alternative<BytecodeStackValue *>(l));
		emit<LoadFast>(
			dst->get_register(), std::get<BytecodeStackValue *>(l)->get_stack_index(), name);
	} break;
	case VariablesResolver::Visibility::CELL:
	case VariablesResolver::Visibility::FREE: {
		ASSERT(m_stack.top().locals.contains(name));
		const auto &l = m_stack.top().locals.at(name);
		ASSERT(std::holds_alternative<BytecodeFreeValue *>(l))
		ASSERT(std::get<BytecodeFreeValue *>(l)->get_name() == name);
		emit<LoadDeref>(
			dst->get_register(), std::get<BytecodeFreeValue *>(l)->get_free_var_index());
	} break;
	}
	return dst;
}

void BytecodeGenerator::delete_var(const std::string &name)
{
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
		TODO();
	} break;
	case VariablesResolver::Visibility::NAME: {
		emit<DeleteName>(load_const(py::String{ name }, m_function_id)->get_index());
	} break;
	case VariablesResolver::Visibility::LOCAL: {
		TODO();
		// emit<DeleteFast>();
	} break;
	case VariablesResolver::Visibility::CELL:
	case VariablesResolver::Visibility::FREE: {
		TODO();
	} break;
	}
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

BytecodeNameValue *BytecodeGenerator::load_name(const std::string &name, size_t function_id)
{
	auto &names = std::next(m_functions.functions.begin(), function_id)->metadata.names;
	for (size_t i = 0; const auto &name_ : names) {
		if (name_ == name) {
			m_values.push_back(std::make_unique<BytecodeNameValue>(name, i));
			return static_cast<BytecodeNameValue *>(m_values.back().get());
		}
		++i;
	}
	names.push_back(name);
	m_values.push_back(std::make_unique<BytecodeNameValue>(name, names.size() - 1));
	return static_cast<BytecodeNameValue *>(m_values.back().get());
}

Value *BytecodeGenerator::visit(const Name *node)
{
	ASSERT(node->ids().size() == 1)
	if (node->context_type() == ContextType::LOAD) {
		return load_var(node->ids()[0]);
	} else if (node->context_type() == ContextType::DELETE) {
		delete_var(node->ids()[0]);
		return nullptr;
	} else {
		TODO();
	}
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
	case BinaryOpType::FLOORDIV: {
		emit<BinaryOperation>(dst->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			BinaryOperation::Operation::FLOORDIV);
	} break;
	case BinaryOpType::MATMUL: {
		emit<BinaryOperation>(dst->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			BinaryOperation::Operation::MATMUL);
	} break;
	case BinaryOpType::LEFTSHIFT: {
		emit<BinaryOperation>(dst->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			BinaryOperation::Operation::LEFTSHIFT);
	} break;
	case BinaryOpType::RIGHTSHIFT: {
		emit<BinaryOperation>(dst->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			BinaryOperation::Operation::RIGHTSHIFT);
	} break;
	case BinaryOpType::AND: {
		emit<BinaryOperation>(dst->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			BinaryOperation::Operation::AND);
	} break;
	case BinaryOpType::OR: {
		emit<BinaryOperation>(dst->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			BinaryOperation::Operation::OR);
	} break;
	case BinaryOpType::XOR: {
		emit<BinaryOperation>(dst->get_register(),
			lhs->get_register(),
			rhs->get_register(),
			BinaryOperation::Operation::XOR);
	} break;
	}

	return dst;
}

template<typename FunctionType>
Value *BytecodeGenerator::generate_function(const FunctionType *node)
{
	std::vector<BytecodeValue *> decorator_functions;
	decorator_functions.reserve(node->decorator_list().size());
	for (const auto &decorator_function : node->decorator_list()) {
		auto *f = generate(decorator_function.get(), m_function_id);
		ASSERT(f)
		decorator_functions.push_back(f);
	}

	std::vector<std::string> varnames;

	m_ctx.push_local_args(node->args());
	const std::string &function_name = Mangler::default_mangler().function_mangle(
		mangle_namespace(m_stack), node->name(), node->source_location());
	auto *f = create_function(function_name);

	const auto &name_visibility_it = m_variable_visibility.find(function_name);
	ASSERT(name_visibility_it != m_variable_visibility.end())
	const auto &name_visibility = name_visibility_it->second->visibility;
	const bool is_generator = name_visibility_it->second->is_generator;

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

	const auto &cellvars = f->function_info().function.metadata.cellvars;
	const size_t arg_count_ = node->args()->argument_names().size()
							  + node->args()->kw_only_argument_names().size()
							  + static_cast<size_t>(node->args()->vararg() != nullptr)
							  + static_cast<size_t>(node->args()->kwarg() != nullptr);
	std::vector<size_t> cell2arg(cellvars.size(), arg_count_);

	create_nested_scope(node->name(), function_name);
	std::vector<std::pair<std::string, BytecodeFreeValue *>> captures;

	for (const auto &arg_name : node->args()->argument_names()) {
		if (auto it = name_visibility.find(arg_name);
			it->second == VariablesResolver::Visibility::CELL) {
			auto *value = create_free_value(arg_name);
			m_stack.top().locals.emplace(arg_name, value);
		}
	}
	for (const auto &arg_name : node->args()->kw_only_argument_names()) {
		if (auto it = name_visibility.find(arg_name);
			it->second == VariablesResolver::Visibility::CELL) {
			auto *value = create_free_value(arg_name);
			m_stack.top().locals.emplace(arg_name, value);
		}
	}
	if (node->args()->vararg()) {
		const auto &arg_name = node->args()->vararg()->name();
		if (auto it = name_visibility.find(arg_name);
			it->second == VariablesResolver::Visibility::CELL) {
			auto *value = create_free_value(arg_name);
			m_stack.top().locals.emplace(arg_name, value);
		}
	}
	if (node->args()->kwarg()) {
		const auto &arg_name = node->args()->kwarg()->name();
		if (auto it = name_visibility.find(arg_name);
			it->second == VariablesResolver::Visibility::CELL) {
			auto *value = create_free_value(arg_name);
			m_stack.top().locals.emplace(arg_name, value);
		}
	}
	for (const auto &capture : m_variable_visibility.at(function_name)->captures) {
		auto *value = create_free_value(capture);
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

	for (size_t idx = 0; const auto &arg_name : node->args()->argument_names()) {
		varnames.push_back(arg_name);
		ASSERT(name_visibility.find(arg_name) != name_visibility.end())
		if (std::find(cellvars.begin(), cellvars.end(), arg_name) != cellvars.end()) {
			ASSERT(m_stack.top().locals.contains(arg_name));
			const auto &l = m_stack.top().locals.at(arg_name);
			ASSERT(std::holds_alternative<BytecodeFreeValue *>(l));
			const size_t cell_idx = std::get<BytecodeFreeValue *>(l)->get_free_var_index();
			cell2arg[cell_idx] = idx;
		}
		idx++;
	}
	for (size_t idx = node->args()->argument_names().size();
		 const auto &arg_name : node->args()->kw_only_argument_names()) {
		varnames.push_back(arg_name);
		ASSERT(name_visibility.find(arg_name) != name_visibility.end())
		if (std::find(cellvars.begin(), cellvars.end(), arg_name) != cellvars.end()) {
			ASSERT(m_stack.top().locals.contains(arg_name));
			const auto &l = m_stack.top().locals.at(arg_name);
			ASSERT(std::holds_alternative<BytecodeFreeValue *>(l));
			const size_t cell_idx = std::get<BytecodeFreeValue *>(l)->get_free_var_index();
			cell2arg[cell_idx] = idx;
		}
		idx++;
	}

	if (node->args()->vararg() != nullptr) {
		const size_t idx =
			node->args()->argument_names().size() + node->args()->kw_only_argument_names().size();
		const auto &arg_name = node->args()->vararg()->name();
		varnames.push_back(arg_name);
		ASSERT(name_visibility.find(arg_name) != name_visibility.end())
		if (std::find(cellvars.begin(), cellvars.end(), arg_name) != cellvars.end()) {
			ASSERT(m_stack.top().locals.contains(arg_name));
			const auto &l = m_stack.top().locals.at(arg_name);
			ASSERT(std::holds_alternative<BytecodeFreeValue *>(l));
			const size_t cell_idx = std::get<BytecodeFreeValue *>(l)->get_free_var_index();
			cell2arg[cell_idx] = idx;
		}
	}

	if (node->args()->kwarg() != nullptr) {
		size_t idx =
			node->args()->argument_names().size() + node->args()->kw_only_argument_names().size();
		if (node->args()->vararg()) { idx++; }
		const auto &arg_name = node->args()->kwarg()->name();
		varnames.push_back(arg_name);
		ASSERT(name_visibility.find(arg_name) != name_visibility.end())
		if (std::find(cellvars.begin(), cellvars.end(), arg_name) != cellvars.end()) {
			ASSERT(m_stack.top().locals.contains(arg_name));
			const auto &l = m_stack.top().locals.at(arg_name);
			ASSERT(std::holds_alternative<BytecodeFreeValue *>(l));
			const size_t cell_idx = std::get<BytecodeFreeValue *>(l)->get_free_var_index();
			cell2arg[cell_idx] = idx;
		}
	}

	set_insert_point(old_block);
	m_ctx.pop_local_args();
	m_stack.pop();
	exit_function(f->function_info().function_id);

	size_t arg_count = node->args()->args().size() + node->args()->posonlyargs().size();
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
					std::get<BytecodeFreeValue *>(value)->get_free_var_index());
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
	if (is_generator) { flags.set(CodeFlags::Flag::GENERATOR); }
	if constexpr (std::is_same_v<FunctionType, AsyncFunctionDefinition>) {
		flags.set(CodeFlags::Flag::COROUTINE);
	}


	// TODO
	// f->function_info().function.metadata.filename = ;
	f->function_info().function.metadata.arg_count = arg_count;
	f->function_info().function.metadata.kwonly_arg_count = kwonly_arg_count;
	f->function_info().function.metadata.cell2arg = std::move(cell2arg);
	f->function_info().function.metadata.nlocals = varnames.size();
	f->function_info().function.metadata.flags = flags;
	f->function_info().function.metadata.varnames = std::move(varnames);

	make_function(f->get_register(), f->get_name(), defaults, kw_defaults, captures_tuple);

	store_name(node->name(), f);
	if (!decorator_functions.empty()) {
		std::vector<BytecodeValue *> args;
		auto *function = load_var(node->name());
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


Value *BytecodeGenerator::visit(const FunctionDefinition *node) { return generate_function(node); }


Value *BytecodeGenerator::visit(const AsyncFunctionDefinition *node)
{
	return generate_function(node);
}

Value *BytecodeGenerator::visit(const Lambda *node)
{
	// TODO: abstract away logic from here and FunctionDefinition* to avoid repetition
	std::vector<std::string> varnames;

	m_ctx.push_local_args(node->args());
	const std::string &function_name = Mangler::default_mangler().function_mangle(
		mangle_namespace(m_stack), "<lambda>", node->source_location());
	auto *f = create_function(function_name);

	const auto &name_visibility_it = m_variable_visibility.find(function_name);
	ASSERT(name_visibility_it != m_variable_visibility.end())
	const auto &name_visibility = name_visibility_it->second->visibility;
	const bool is_generator = name_visibility_it->second->is_generator;

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

	const auto &cellvars = f->function_info().function.metadata.cellvars;
	const size_t arg_count_ = node->args()->argument_names().size()
							  + node->args()->kw_only_argument_names().size()
							  + node->args()->kw_only_argument_names().size()
							  + static_cast<size_t>(node->args()->vararg() != nullptr)
							  + static_cast<size_t>(node->args()->kwarg() != nullptr);
	std::vector<size_t> cell2arg(cellvars.size(), arg_count_);

	create_nested_scope("<lambda>", function_name);
	std::vector<std::pair<std::string, BytecodeFreeValue *>> captures;

	for (const auto &arg_name : node->args()->argument_names()) {
		if (auto it = name_visibility.find(arg_name);
			it->second == VariablesResolver::Visibility::CELL) {
			auto *value = create_free_value(arg_name);
			m_stack.top().locals.emplace(arg_name, value);
		}
	}
	for (const auto &arg_name : node->args()->kw_only_argument_names()) {
		if (auto it = name_visibility.find(arg_name);
			it->second == VariablesResolver::Visibility::CELL) {
			auto *value = create_free_value(arg_name);
			m_stack.top().locals.emplace(arg_name, value);
		}
	}
	if (node->args()->vararg()) {
		const auto &arg_name = node->args()->vararg()->name();
		if (auto it = name_visibility.find(arg_name);
			it->second == VariablesResolver::Visibility::CELL) {
			auto *value = create_free_value(arg_name);
			m_stack.top().locals.emplace(arg_name, value);
		}
	}
	if (node->args()->kwarg()) {
		const auto &arg_name = node->args()->kwarg()->name();
		if (auto it = name_visibility.find(arg_name);
			it->second == VariablesResolver::Visibility::CELL) {
			auto *value = create_free_value(arg_name);
			m_stack.top().locals.emplace(arg_name, value);
		}
	}
	for (const auto &capture : m_variable_visibility.at(function_name)->captures) {
		auto *value = create_free_value(capture);
		captures.emplace_back(capture, value);
		m_stack.top().locals.emplace(capture, value);
	}

	auto *block = allocate_block(f->function_info().function_id);
	auto *old_block = m_current_block;
	set_insert_point(block);

	generate(node->args().get(), f->function_info().function_id);

	auto *lambda_return_value = generate(node->body().get(), f->function_info().function_id);
	ASSERT(lambda_return_value);
	emit<ReturnValue>(lambda_return_value->get_register());

	// always return None
	// this can be optimised away later on
	auto none_value_register = allocate_register();
	auto *value = load_const(py::NameConstant{ py::NoneType{} }, f->function_info().function_id);
	emit<LoadConst>(none_value_register, value->get_index());
	emit<ReturnValue>(none_value_register);

	for (size_t idx = 0; const auto &arg_name : node->args()->argument_names()) {
		varnames.push_back(arg_name);
		ASSERT(name_visibility.find(arg_name) != name_visibility.end())
		if (std::find(cellvars.begin(), cellvars.end(), arg_name) != cellvars.end()) {
			ASSERT(m_stack.top().locals.contains(arg_name));
			const auto &l = m_stack.top().locals.at(arg_name);
			ASSERT(std::holds_alternative<BytecodeFreeValue *>(l));
			const size_t cell_idx = std::get<BytecodeFreeValue *>(l)->get_free_var_index();
			cell2arg[cell_idx] = idx;
		}
		idx++;
	}
	for (size_t idx = node->args()->argument_names().size();
		 const auto &arg_name : node->args()->kw_only_argument_names()) {
		varnames.push_back(arg_name);
		ASSERT(name_visibility.find(arg_name) != name_visibility.end())
		if (std::find(cellvars.begin(), cellvars.end(), arg_name) != cellvars.end()) {
			ASSERT(m_stack.top().locals.contains(arg_name));
			const auto &l = m_stack.top().locals.at(arg_name);
			ASSERT(std::holds_alternative<BytecodeFreeValue *>(l));
			const size_t cell_idx = std::get<BytecodeFreeValue *>(l)->get_free_var_index();
			cell2arg[cell_idx] = idx;
		}
		idx++;
	}

	if (node->args()->vararg() != nullptr) {
		const size_t idx =
			node->args()->argument_names().size() + node->args()->kw_only_argument_names().size();
		const auto &arg_name = node->args()->vararg()->name();
		varnames.push_back(arg_name);
		ASSERT(name_visibility.find(arg_name) != name_visibility.end())
		if (std::find(cellvars.begin(), cellvars.end(), arg_name) != cellvars.end()) {
			ASSERT(m_stack.top().locals.contains(arg_name));
			const auto &l = m_stack.top().locals.at(arg_name);
			ASSERT(std::holds_alternative<BytecodeFreeValue *>(l));
			const size_t cell_idx = std::get<BytecodeFreeValue *>(l)->get_free_var_index();
			cell2arg[cell_idx] = idx;
		}
	}

	if (node->args()->kwarg() != nullptr) {
		size_t idx =
			node->args()->argument_names().size() + node->args()->kw_only_argument_names().size();
		if (node->args()->vararg()) { idx++; }
		const auto &arg_name = node->args()->kwarg()->name();
		varnames.push_back(arg_name);
		ASSERT(name_visibility.find(arg_name) != name_visibility.end())
		if (std::find(cellvars.begin(), cellvars.end(), arg_name) != cellvars.end()) {
			ASSERT(m_stack.top().locals.contains(arg_name));
			const auto &l = m_stack.top().locals.at(arg_name);
			ASSERT(std::holds_alternative<BytecodeFreeValue *>(l));
			const size_t cell_idx = std::get<BytecodeFreeValue *>(l)->get_free_var_index();
			cell2arg[cell_idx] = idx;
		}
	}

	set_insert_point(old_block);
	m_ctx.pop_local_args();
	m_stack.pop();
	exit_function(f->function_info().function_id);

	size_t arg_count = node->args()->args().size() + node->args()->posonlyargs().size();
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
					std::get<BytecodeFreeValue *>(value)->get_free_var_index());
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
	if (is_generator) { flags.set(CodeFlags::Flag::GENERATOR); }


	// TODO
	// f->function_info().function.metadata.filename = ;
	f->function_info().function.metadata.arg_count = arg_count;
	f->function_info().function.metadata.kwonly_arg_count = kwonly_arg_count;
	f->function_info().function.metadata.cell2arg = std::move(cell2arg);
	f->function_info().function.metadata.nlocals = varnames.size();
	f->function_info().function.metadata.flags = flags;
	f->function_info().function.metadata.varnames = std::move(varnames);

	make_function(f->get_register(), f->get_name(), defaults, kw_defaults, captures_tuple);

	return f;
}

Value *BytecodeGenerator::visit(const Arguments *node)
{
	for (const auto &arg : node->posonlyargs()) { generate(arg.get(), m_function_id); }
	for (const auto &arg : node->args()) { generate(arg.get(), m_function_id); }
	for (const auto &arg : node->kwonlyargs()) { generate(arg.get(), m_function_id); }
	if (node->vararg()) { generate(node->vararg().get(), m_function_id); }
	if (node->kwarg()) { generate(node->kwarg().get(), m_function_id); }

	return nullptr;
}

Value *BytecodeGenerator::visit(const Argument *node)
{
	const auto &var_scope =
		m_variable_visibility.at(m_stack.top().mangled_name)->visibility.at(node->name());
	switch (var_scope) {
	case VariablesResolver::Visibility::CELL: {
		m_stack.top().locals.emplace(node->name(), create_free_value(node->name()));
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
	auto *src = [&]() -> BytecodeValue * {
		if (node->value()) {
			return generate(node->value().get(), m_function_id);
		} else {
			auto *none_value = create_value();
			auto *value = load_const(py::NameConstant{ py::NoneType{} }, m_function_id);
			emit<LoadConst>(none_value->get_register(), value->get_index());
			return none_value;
		}
	}();
	if (m_clear_exception_before_return_functions.contains(m_function_id)) {
		emit<ClearExceptionState>();
	}

	// if (auto it = m_current_exception_depth.find(m_function_id);
	// 	it != m_current_exception_depth.end()) {
	// 	size_t depth = it->second;
	// 	while (depth-- > 0) { emit<LeaveExceptionHandling>(); }
	// }

	const auto transforms = m_return_transform[m_function_id];
	if (!transforms.empty()) {
		m_return_transform[m_function_id].pop_back();
		transforms.back()(true);
		std::for_each(transforms.rbegin() + 1, transforms.rend(), [this](const auto &f) {
			m_return_transform[m_function_id].pop_back();
			f(false);
		});
	}
	m_return_transform[m_function_id] = std::move(transforms);
	emit<ReturnValue>(src->get_register());
	return src;
}

Value *BytecodeGenerator::visit(const Yield *node)
{
	auto *src = generate(node->value().get(), m_function_id);
	ASSERT(src);
	emit<YieldValue>(src->get_register());
	auto *bidirectional_value = create_value();
	emit<YieldLoad>(bidirectional_value->get_register());
	return bidirectional_value;
}

Value *BytecodeGenerator::visit(const ast::YieldFrom *node)
{
	auto *src = generate(node->value().get(), m_function_id);
	ASSERT(src);
	auto *iterator = create_value();
	emit<GetYieldFromIter>(iterator->get_register(), src->get_register());
	auto *none_static = load_const(py::NameConstant{ py::NoneType{} }, m_function_id);
	auto *none = create_value();
	emit<LoadConst>(none->get_register(), none_static->get_index());
	auto *result = create_value();
	emit<::YieldFrom>(result->get_register(), iterator->get_register(), none->get_register());
	return result;
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
			emit<StoreAttr>(dst->get_register(),
				src->get_register(),
				load_name(ast_attr->attr(), m_function_id)->get_index());
		} else if (auto ast_tuple = as<Tuple>(target)) {
			std::vector<BytecodeValue *> unpacked_values;
			std::vector<Register> unpacked_registers;
			for (size_t i = 0; i < ast_tuple->elements().size(); ++i) {
				auto &unpacked_value = unpacked_values.emplace_back(create_value());
				unpacked_registers.push_back(unpacked_value->get_register());
			}
			emit<UnpackSequence>(unpacked_registers, src->get_register());
			for (size_t idx = 0; const auto &unpacked_value : unpacked_values) {
				const auto &el = ast_tuple->elements()[idx++];
				if (auto name = as<Name>(el)) {
					store_name(name->ids()[0], unpacked_value);
				} else if (auto attr = as<Attribute>(el)) {
					auto *dst_obj = generate(attr->value().get(), m_function_id);
					emit<StoreAttr>(dst_obj->get_register(),
						unpacked_value->get_register(),
						load_name(attr->attr(), m_function_id)->get_index());
				} else if (auto subscript = as<Subscript>(el)) {
					auto *dst_obj = generate(subscript->value().get(), m_function_id);
					const auto &slice = subscript->slice();
					const auto *index = build_slice(slice);
					emit<StoreSubscript>(dst_obj->get_register(),
						index->get_register(),
						unpacked_value->get_register());
				} else {
					TODO();
				}
			}
		} else if (auto ast_subscript = as<Subscript>(target)) {
			auto *obj = generate(ast_subscript->value().get(), m_function_id);
			const auto &slice = ast_subscript->slice();
			const auto *index = build_slice(slice);
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
	std::vector<Register> keywords;

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
		if (!node->args().empty()) { requires_args_expansion = true; }
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
			std::vector<std::optional<Register>> key_registers;
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
			keywords.push_back(load_name(*keyword_argname, m_function_id)->get_index());
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
			emit<MethodCall>(func->get_register(), std::move(arg_registers));
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
	auto end_label = make_label(fmt::format("IF_END_{}", if_count++), m_function_id);

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
	auto previous_start_label = m_ctx.set_current_loop_start_label(forloop_start_label);
	auto previous_end_label = m_ctx.set_current_loop_end_label(forloop_end_label);
	// call the __next__ implementation
	emit<ForIter>(iter_variable->get_register(), iterator_register, forloop_end_label);

	if (auto target = as<Name>(node->target())) {
		auto target_ids = target->ids();
		if (target_ids.size() != 1) { TODO(); }
		auto target_name = target_ids[0];
		store_name(target_name, iter_variable);
	} else if (auto target = as<Tuple>(node->target())) {
		std::vector<Register> dst;
		std::vector<BytecodeValue *> values;
		std::vector<std::string> names;
		dst.reserve(target->elements().size());
		values.reserve(target->elements().size());
		names.reserve(target->elements().size());

		for (const auto &el : target->elements()) {
			ASSERT(el->node_type() == ASTNodeType::Name);
			ASSERT(as<Name>(el)->ids().size() == 1);

			names.push_back(as<Name>(el)->ids()[0]);
			values.push_back(create_value(names.back()));
			dst.push_back(values.back()->get_register());
		}
		emit<UnpackSequence>(dst, iter_variable->get_register());

		for (auto name_it = names.begin(); const auto &v : values) {
			store_name(*name_it, v);
			++name_it;
		}
	} else {
		TODO();
	}

	// body
	for (const auto &el : node->body()) { generate(el.get(), m_function_id); }
	emit<Jump>(forloop_start_label);

	// orelse
	for (const auto &el : node->orelse()) { generate(el.get(), m_function_id); }
	bind(*forloop_end_label);

	m_ctx.set_current_loop_start_label(previous_start_label);
	m_ctx.set_current_loop_end_label(previous_end_label);

	return nullptr;
}

Value *BytecodeGenerator::visit(const Continue *)
{
	emit<Jump>(m_ctx.get_current_loop_start_label());
	return nullptr;
}

Value *BytecodeGenerator::visit(const Break *)
{
	emit<Jump>(m_ctx.get_current_loop_end_label());
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
	auto previous_start_label = m_ctx.set_current_loop_start_label(while_loop_start_label);
	auto previous_end_label = m_ctx.set_current_loop_end_label(while_loop_end_label);

	const auto *test_result = generate(node->test().get(), m_function_id);
	emit<JumpIfFalse>(test_result->get_register(), while_loop_end_label);

	// body
	for (const auto &el : node->body()) { generate(el.get(), m_function_id); }
	emit<Jump>(while_loop_start_label);

	// orelse
	bind(*while_loop_end_label);
	for (const auto &el : node->orelse()) { generate(el.get(), m_function_id); }

	m_ctx.set_current_loop_start_label(previous_start_label);
	m_ctx.set_current_loop_start_label(previous_end_label);

	return nullptr;
}

Value *BytecodeGenerator::visit(const Compare *node)
{
	const auto *lhs = generate(node->lhs().get(), m_function_id);
	const auto &comparators = node->comparators();
	const auto &ops = node->ops();
	BytecodeValue *result{ nullptr };

	for (size_t idx = 0; idx < comparators.size(); ++idx) {
		const auto *rhs = generate(comparators[idx].get(), m_function_id);
		const auto op = ops[idx];
		result = create_value();

		switch (op) {
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
		lhs = rhs;
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

Value *BytecodeGenerator::visit(const Set *node)
{
	std::vector<Register> element_registers;
	element_registers.reserve(node->elements().size());

	for (const auto &el : node->elements()) {
		auto *element_value = generate(el.get(), m_function_id);
		element_registers.push_back(element_value->get_register());
	}

	return build_set(element_registers);
}

Value *BytecodeGenerator::visit(const ClassDefinition *node)
{
	if (!node->decorator_list().empty()) { TODO(); }
	auto class_mangled_name = Mangler::default_mangler().class_mangle(
		mangle_namespace(m_stack), node->name(), node->source_location());

	const auto &name_visibility_it = m_variable_visibility.find(class_mangled_name);
	ASSERT(name_visibility_it != m_variable_visibility.end())
	const auto &class_scope = name_visibility_it->second;

	auto *class_builder_func = create_function(class_mangled_name);
	create_nested_scope(node->name(), class_mangled_name);
	size_t class_id = class_builder_func->function_info().function_id;

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

	if (class_scope->requires_class_ref) {
		auto *__class__ = create_free_value("__class__");
		ASSERT(__class__->get_free_var_index() == 0);
		m_stack.top().locals.emplace("__class__", __class__);
	}

	// the actual class definition
	for (const auto &el : node->body()) { generate(el.get(), class_id); }

	if (class_scope->requires_class_ref) {
		auto it = m_stack.top().locals.find("__class__");
		ASSERT(it != m_stack.top().locals.end());
		ASSERT(std::holds_alternative<BytecodeFreeValue *>(it->second));
		auto *__class__ = std::get<BytecodeFreeValue *>(it->second);
		auto *class_value = create_value();
		emit<LoadClosure>(class_value->get_register(), __class__->get_free_var_index());
		emit<StoreName>("__classcell__", class_value->get_register());
		emit<ReturnValue>(class_value->get_register());
	} else {
		emit<LoadConst>(return_none_register,
			load_const(py::NameConstant{ py::NoneType{} }, class_id)->get_index());
		emit<ReturnValue>(return_none_register);
	}

	m_stack.pop();
	exit_function(class_builder_func->function_info().function_id);

	set_insert_point(old_block);

	class_builder_func->function_info().function.metadata.arg_count = 0;
	class_builder_func->function_info().function.metadata.kwonly_arg_count = 0;
	class_builder_func->function_info().function.metadata.cell2arg = {};
	class_builder_func->function_info().function.metadata.nlocals = 0;
	class_builder_func->function_info().function.metadata.flags = CodeFlags::create();
	class_builder_func->function_info().function.metadata.cellvars =
		class_scope->requires_class_ref ? std::vector<std::string>{ "__class__" }
										: std::vector<std::string>{};

	std::vector<Register> arg_registers;
	arg_registers.reserve(2 + node->bases().size());
	std::vector<Register> kwarg_registers;
	std::vector<Register> keyword_names;

	const auto builtin_build_class_register = allocate_register();
	const auto class_name_register = allocate_register();

	arg_registers.push_back(class_builder_func->get_register());
	arg_registers.push_back(class_name_register);

	for (const auto &base : node->bases()) {
		auto *base_value = generate(base.get(), m_function_id);
		arg_registers.push_back(base_value->get_register());
	}
	for (const auto &keyword : node->keywords()) {
		auto *kw_value = generate(keyword.get(), m_function_id);
		kwarg_registers.push_back(kw_value->get_register());
		if (!keyword->arg().has_value()) { TODO(); }
		keyword_names.push_back(load_name(*keyword->arg(), m_function_id)->get_index());
	}

	emit<LoadBuildClass>(builtin_build_class_register);
	emit<LoadConst>(class_name_register,
		load_const(py::String{ class_mangled_name }, m_function_id)->get_index());
	make_function(class_builder_func->get_register(), class_mangled_name, {}, {}, {});

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

	std::vector<std::optional<Register>> key_registers;
	std::vector<Register> value_registers;

	for (const auto &key : node->keys()) {
		if (key) {
			auto *key_value = generate(key.get(), m_function_id);
			key_registers.emplace_back(key_value->get_register());
		} else {
			key_registers.push_back(std::nullopt);
		}
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
		emit<LoadMethod>(method_name->get_register(),
			this_value->get_register(),
			load_name(node->attr(), m_function_id)->get_index());
		return method_name;
	} else if (node->context() == ContextType::LOAD) {
		auto *attribute_value = create_value();
		const auto &attr_name = load_name(node->attr(), m_function_id);
		emit<LoadAttr>(
			attribute_value->get_register(), this_value->get_register(), attr_name->get_index());
		return attribute_value;
	} else if (node->context() == ContextType::STORE) {
		auto *attribute_value = create_value();
		const auto &attr_name = load_name(node->attr(), m_function_id);
		emit<LoadAttr>(
			attribute_value->get_register(), this_value->get_register(), attr_name->get_index());
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
	auto *lhs = [this, node]() {
		if (auto named_target = as<Name>(node->target())) {
			ASSERT(named_target->context_type() == ContextType::STORE)
			if (named_target->ids().size() != 1) { TODO(); }
			return load_var(named_target->ids()[0]);
		} else if (auto attr = as<Attribute>(node->target())) {
			auto *r = generate(attr.get(), m_function_id);
			ASSERT(r);
			return r;
		} else {
			TODO();
		}
	}();

	const auto *rhs = generate(node->value().get(), m_function_id);
	switch (node->op()) {
	case BinaryOpType::PLUS: {
		emit<InplaceOp>(lhs->get_register(), rhs->get_register(), InplaceOp::Operation::PLUS);
	} break;
	case BinaryOpType::MINUS: {
		emit<InplaceOp>(lhs->get_register(), rhs->get_register(), InplaceOp::Operation::MINUS);
	} break;
	case BinaryOpType::MULTIPLY: {
		emit<InplaceOp>(lhs->get_register(), rhs->get_register(), InplaceOp::Operation::MULTIPLY);
	} break;
	case BinaryOpType::EXP: {
		emit<InplaceOp>(lhs->get_register(), rhs->get_register(), InplaceOp::Operation::EXP);
	} break;
	case BinaryOpType::MODULO: {
		emit<InplaceOp>(lhs->get_register(), rhs->get_register(), InplaceOp::Operation::MODULO);
	} break;
	case BinaryOpType::SLASH: {
		emit<InplaceOp>(lhs->get_register(), rhs->get_register(), InplaceOp::Operation::SLASH);
	}
	case BinaryOpType::FLOORDIV: {
		emit<InplaceOp>(lhs->get_register(), rhs->get_register(), InplaceOp::Operation::FLOORDIV);
	}
	case BinaryOpType::MATMUL: {
		emit<InplaceOp>(lhs->get_register(), rhs->get_register(), InplaceOp::Operation::MATMUL);
	}
	case BinaryOpType::LEFTSHIFT: {
		emit<InplaceOp>(lhs->get_register(), rhs->get_register(), InplaceOp::Operation::LEFTSHIFT);
	} break;
	case BinaryOpType::RIGHTSHIFT: {
		emit<InplaceOp>(lhs->get_register(), rhs->get_register(), InplaceOp::Operation::RIGHTSHIFT);
	} break;
	case ast::BinaryOpType::AND: {
		emit<InplaceOp>(lhs->get_register(), rhs->get_register(), InplaceOp::Operation::AND);
	} break;
	case ast::BinaryOpType::OR: {
		emit<InplaceOp>(lhs->get_register(), rhs->get_register(), InplaceOp::Operation::OR);
	} break;
	case ast::BinaryOpType::XOR: {
		emit<InplaceOp>(lhs->get_register(), rhs->get_register(), InplaceOp::Operation::XOR);
	} break;
	}

	if (auto named_target = as<Name>(node->target())) {
		if (named_target->ids().size() != 1) { TODO(); }
		store_name(named_target->ids()[0], lhs);
	} else if (auto attr = as<Attribute>(node->target())) {
		auto *obj = generate(attr->value().get(), m_function_id);
		emit<StoreAttr>(obj->get_register(),
			lhs->get_register(),
			load_name(attr->attr(), m_function_id)->get_index());
	} else {
		TODO();
	}

	return lhs;
}

Value *BytecodeGenerator::visit(const Import *node)
{
	for (const auto &n : node->names()) {
		auto *name = load_name(n.name, m_function_id);
		auto *from_list = load_const(py::NameConstant{ py::NoneType{} }, m_function_id);
		auto *level = load_const(py::Number{ int64_t{ 0 } }, m_function_id);

		auto *from_list_value = create_value();
		auto *level_value = create_value();

		emit<LoadConst>(from_list_value->get_register(), from_list->get_index());
		emit<LoadConst>(level_value->get_register(), level->get_index());

		auto *module_value = create_value();
		emit<ImportName>(module_value->get_register(),
			name->get_index(),
			from_list_value->get_register(),
			level_value->get_register());

		if (!n.asname.empty()) {
			store_name(n.asname, module_value);
		} else {
			if (const auto idx = n.name.find('.'); idx != std::string::npos) {
				const auto varname = n.name.substr(0, idx);
				store_name(varname, module_value);
			} else {
				store_name(n.name, module_value);
			}
		}
	}
	return nullptr;
}

Value *BytecodeGenerator::visit(const ast::ImportFrom *node)
{
	std::vector<Register> names;
	for (const auto &n : node->names()) {
		auto name = load_const(py::String{ n.name }, m_function_id);
		auto name_value = create_value();
		emit<LoadConst>(name_value->get_register(), name->get_index());
		names.push_back(name_value->get_register());
	}

	auto *name = load_name(node->module(), m_function_id);
	auto *from_list_value = build_tuple(names);
	auto *level = load_const(py::Number{ static_cast<int64_t>(node->level()) }, m_function_id);

	auto *level_value = create_value();

	emit<LoadConst>(level_value->get_register(), level->get_index());

	auto *module_value = create_value();
	emit<ImportName>(module_value->get_register(),
		name->get_index(),
		from_list_value->get_register(),
		level_value->get_register());

	for (const auto &n : node->names()) {
		if (n.name == "*") {
			emit<ImportStar>(module_value->get_register());
		} else {
			auto *imported_object = create_value();
			auto *name = load_name(n.name, m_function_id);
			emit<::ImportFrom>(
				imported_object->get_register(), name->get_index(), module_value->get_register());
			if (n.asname.empty()) {
				store_name(n.name, imported_object);
			} else {
				store_name(n.asname, imported_object);
			}
		}
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

BytecodeValue *BytecodeGenerator::build_slice(const ast::Subscript::SliceType &sliceNode)
{
	if (std::holds_alternative<Subscript::Index>(sliceNode)) {
		return generate(std::get<Subscript::Index>(sliceNode).value.get(), m_function_id);
	} else if (std::holds_alternative<Subscript::Slice>(sliceNode)) {
		const auto &slice = std::get<Subscript::Slice>(sliceNode);
		auto *index = create_value();
		auto *lower = slice.lower ? generate(slice.lower.get(), m_function_id) : nullptr;
		auto *upper = slice.upper ? generate(slice.upper.get(), m_function_id) : nullptr;
		auto *step = slice.step ? generate(slice.step.get(), m_function_id) : nullptr;
		if (!lower && !upper && !step) {
			auto *none = load_const(py::NameConstant{ py::NoneType{} }, m_function_id);
			auto *none_value = create_value();
			emit<LoadConst>(none_value->get_register(), none->get_index());
			emit<BuildSlice>(
				index->get_register(), none_value->get_register(), none_value->get_register());
		} else if (!upper && !step) {
			auto *none = load_const(py::NameConstant{ py::NoneType{} }, m_function_id);
			auto *none_value = create_value();
			emit<LoadConst>(none_value->get_register(), none->get_index());
			emit<BuildSlice>(
				index->get_register(), lower->get_register(), none_value->get_register());
		} else if (!step) {
			if (!lower) {
				auto *none = load_const(py::NameConstant{ py::NoneType{} }, m_function_id);
				lower = create_value();
				emit<LoadConst>(lower->get_register(), none->get_index());
			}
			emit<BuildSlice>(index->get_register(), lower->get_register(), upper->get_register());
		} else {
			if (!lower) {
				auto *none = load_const(py::NameConstant{ py::NoneType{} }, m_function_id);
				lower = create_value();
				emit<LoadConst>(lower->get_register(), none->get_index());
			}
			if (!upper) {
				auto *none = load_const(py::NameConstant{ py::NoneType{} }, m_function_id);
				upper = create_value();
				emit<LoadConst>(upper->get_register(), none->get_index());
			}
			emit<BuildSlice>(index->get_register(),
				lower->get_register(),
				upper->get_register(),
				step->get_register());
		}
		return index;
	} else if (std::holds_alternative<Subscript::ExtSlice>(sliceNode)) {
		TODO();
	}
	TODO();
	return nullptr;
}

Value *BytecodeGenerator::visit(const Subscript *node)
{
	auto *result = create_value();
	const auto *value = generate(node->value().get(), m_function_id);
	const auto *index = build_slice(node->slice());

	switch (node->context()) {
	case ContextType::DELETE: {
		emit<DeleteSubscript>(value->get_register(), index->get_register());
	} break;
	case ContextType::LOAD: {
		emit<BinarySubscript>(result->get_register(), value->get_register(), index->get_register());
	} break;
	case ContextType::STORE: {
		// handled in Assign
		TODO();
	} break;
	case ContextType::UNSET: {
		TODO();
	} break;
	}
	return result;
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
	auto cleanup_label =
		make_label(fmt::format("WITH_CLEANUP_{}", exit_label_count++), m_function_id);

	auto *body_block = allocate_block(m_function_id);

	if (node->items().size() > 1) { TODO(); }
	std::vector<BytecodeValue *> with_item_results;

	for (const auto &item : node->items()) {
		with_item_results.push_back(generate(item.get(), m_function_id));
	}

	emit<SetupWith>(cleanup_label);

	set_insert_point(body_block);

	auto with_exit_factory = [this, &with_item_results](bool first) {
		auto exit_label = make_label(fmt::format("WITH_EXIT_{}", exit_label_count), m_function_id);
		for (const auto &item : with_item_results) {
			auto *exit_result = create_value();
			auto *exit_method = create_value();

			if (!first) emit<LeaveExceptionHandling>();

			// the result of the call to __exit__ is stored in the return register (r0)
			// so we need to save the current value of r0 and restore it after calling the
			// method
			auto *maybe_return_value = create_value();
			auto *last_call_return_value = create_return_value();
			emit<Move>(maybe_return_value->get_register(), last_call_return_value->get_register());
			emit<LoadMethod>(exit_method->get_register(),
				item->get_register(),
				load_name("__exit__", m_function_id)->get_index());
			emit<WithExceptStart>(exit_result->get_register(), exit_method->get_register());
			emit<Move>(last_call_return_value->get_register(), maybe_return_value->get_register());
			emit<JumpIfTrue>(exit_result->get_register(), exit_label);
		}
		emit<ReRaise>();
		bind(*exit_label);
		emit<ClearExceptionState>();
	};

	{
		ScopedWithStatement scope{ *this, with_exit_factory, m_function_id };

		for (const auto &statement : node->body()) { generate(statement.get(), m_function_id); }
		emit<LeaveExceptionHandling>();
		auto *cleanup_block = allocate_block(m_function_id);
		set_insert_point(cleanup_block);
		bind(*cleanup_label);
		with_exit_factory(true);

		auto *next_block = allocate_block(m_function_id);
		set_insert_point(next_block);
	}

	return nullptr;
}

Value *BytecodeGenerator::visit(const WithItem *node)
{
	auto *ctx_expr_result = generate(node->context_expr().get(), m_function_id);
	auto *enter_method = create_value();
	auto *ctx_expr = create_value();
	emit<Move>(ctx_expr->get_register(), ctx_expr_result->get_register());
	emit<LoadMethod>(enter_method->get_register(),
		ctx_expr->get_register(),
		load_name("__enter__", m_function_id)->get_index());
	emit<MethodCall>(enter_method->get_register(), std::vector<Register>{});
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

Value *BytecodeGenerator::visit(const Try *node)
{
	static size_t try_op_count = 0;
	size_t exception_count = 0;

	auto next_exception_label = make_label(
		fmt::format("TRY_EXC_COUNT_{}_{}", try_op_count, exception_count++), m_function_id);
	auto orelse_label =
		node->orelse().empty()
			? nullptr
			: make_label(fmt::format("TRY_ORELSE_COUNT_{}", try_op_count), m_function_id);
	auto finally_label =
		make_label(fmt::format("TRY_FINALLY_OP_COUNT_{}", try_op_count++), m_function_id);

	emit<SetupExceptionHandling>(next_exception_label);

	auto *body_block = allocate_block(m_function_id);
	set_insert_point(body_block);

	auto finally_code_with_exception = [this, &node](bool first) {
		if (!first) emit<LeaveExceptionHandling>();
		if (node->finalbody().empty()) {
			auto *empty_finally_block = allocate_block(m_function_id);
			set_insert_point(empty_finally_block);
		} else {
			auto *finally_block_with_reraise = allocate_block(m_function_id);
			set_insert_point(finally_block_with_reraise);
			{
				for (const auto &statement : node->finalbody()) {
					generate(statement.get(), m_function_id);
				}
			}
		}
		auto *next_block = allocate_block(m_function_id);
		set_insert_point(next_block);
	};

	{
		ScopedTryStatement try_scope{ *this, finally_code_with_exception, m_function_id };

		for (const auto &statement : node->body()) { generate(statement.get(), m_function_id); }

		emit<LeaveExceptionHandling>();

		if (!node->orelse().empty()) {
			ASSERT(orelse_label)
			emit<Jump>(orelse_label);
		} else {
			emit<Jump>(finally_label);
		}

		const size_t exception_depth = m_current_exception_depth[m_function_id];

		for (const auto &handler : node->handlers()) {
			auto *exception_handler_block = allocate_block(m_function_id);
			set_insert_point(exception_handler_block);
			bind(*next_exception_label);
			if (!handler->type()) {
				if (handler != *(node->handlers().end() - 1)) {
					// FIXME: implement SyntaxError and error throwing when parsing source code
					spdlog::error("SyntaxError: default 'except:' must be last");
					std::abort();
				}
				next_exception_label = nullptr;
			} else {
				next_exception_label =
					make_label(fmt::format("TRY_EXC_COUNT_{}_{}", try_op_count, exception_count++),
						m_function_id);
				auto *exception_type = generate(handler->type().get(), m_function_id);
				emit<JumpIfNotExceptionMatch>(exception_type->get_register(), next_exception_label);
			}
			auto *exception_handler_body = allocate_block(m_function_id);
			set_insert_point(exception_handler_body);
			{
				ScopedClearExceptionBeforeReturn s{ *this, m_function_id };
				// emit<LeaveExceptionHandling>();
				m_current_exception_depth[m_function_id] = exception_depth - 1;
				for (const auto &el : handler->body()) { generate(el.get(), m_function_id); }
				m_current_exception_depth[m_function_id] = exception_depth;
				emit<ClearExceptionState>();
			}
			emit<Jump>(finally_label);
		}

		if (!node->orelse().empty()) {
			bind(*orelse_label);
			for (const auto &statement : node->orelse()) {
				generate(statement.get(), m_function_id);
			}
			emit<Jump>(finally_label);
		}
	}

	if (next_exception_label) bind(*next_exception_label);

	// emit<LeaveExceptionHandling>();

	if (node->finalbody().empty()) {
		auto *empty_finally_block = allocate_block(m_function_id);
		set_insert_point(empty_finally_block);
		emit<ReRaise>();
		bind(*finally_label);
	} else {
		auto *finally_block_with_reraise = allocate_block(m_function_id);
		set_insert_point(finally_block_with_reraise);
		{
			ScopedClearExceptionBeforeReturn s{ *this, m_function_id };
			for (const auto &statement : node->finalbody()) {
				generate(statement.get(), m_function_id);
			}
		}
		emit<ReRaise>();

		bind(*finally_label);
		auto *finally_block = allocate_block(m_function_id);
		set_insert_point(finally_block);
		for (const auto &statement : node->finalbody()) {
			generate(statement.get(), m_function_id);
		}
		// emit<ClearExceptionState>();
	}
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

Value *BytecodeGenerator::visit(const NonLocal *) { return nullptr; }

Value *BytecodeGenerator::visit(const Delete *node)
{
	for (const auto &target : node->targets()) { generate(target.get(), m_function_id); }
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
	auto end_label = make_label(fmt::format("ASSERT_END_{}", assert_count++), m_function_id);

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

Value *BytecodeGenerator::visit(const Comprehension *) { TODO(); }

std::
	tuple<std::vector<std::shared_ptr<Label>>, std::vector<std::shared_ptr<Label>>, BytecodeValue *>
	BytecodeGenerator::visit_comprehension(
		const std::vector<std::shared_ptr<Comprehension>> &comprehensions,
		std::function<BytecodeValue *()> container_builder)
{
	static size_t comprehension_count = 0;

	std::vector<std::shared_ptr<Label>> start_labels;
	std::vector<std::shared_ptr<Label>> end_labels;

	auto *container = container_builder();

	auto *src = create_stack_value();
	auto *it = create_value();
	emit<LoadFast>(it->get_register(), src->get_stack_index(), ".0");

	for (bool first = true; const auto &comprehension : comprehensions) {
		auto *node = comprehension.get();
		auto start_label =
			make_label(fmt::format("COMPREHENSION_START_{}", comprehension_count), m_function_id);
		auto end_label =
			make_label(fmt::format("COMPREHENSION_END_{}", comprehension_count++), m_function_id);

		if (!first) {
			auto iterable = generate(comprehension->iter().get(), m_function_id);
			it = create_value();
			emit<GetIter>(it->get_register(), iterable->get_register());
		}

		auto *dst = create_value();
		bind(*start_label);
		emit<ForIter>(dst->get_register(), it->get_register(), end_label);
		if (node->target()->node_type() == ASTNodeType::Name) {
			const auto name = std::static_pointer_cast<Name>(node->target());
			ASSERT(name->ids().size() == 1)
			store_name(name->ids()[0], dst);
		} else if (auto target = as<Tuple>(node->target())) {
			std::vector<Register> unpack_dst;
			std::vector<BytecodeValue *> values;
			std::vector<std::string> names;
			unpack_dst.reserve(target->elements().size());
			values.reserve(target->elements().size());
			names.reserve(target->elements().size());

			for (const auto &el : target->elements()) {
				ASSERT(el->node_type() == ASTNodeType::Name);
				ASSERT(as<Name>(el)->ids().size() == 1);

				names.push_back(as<Name>(el)->ids()[0]);
				values.push_back(create_value(names.back()));
				unpack_dst.push_back(values.back()->get_register());
			}
			emit<UnpackSequence>(unpack_dst, dst->get_register());

			for (auto name_it = names.begin(); const auto &v : values) {
				store_name(*name_it, v);
				++name_it;
			}
		} else {
			TODO();
		}

		for (const auto &if_ : node->ifs()) {
			auto *result = generate(if_.get(), m_function_id);
			ASSERT(result);
			emit<JumpIfFalse>(result->get_register(), start_label);
		}

		start_labels.push_back(std::move(start_label));
		end_labels.push_back(std::move(end_label));

		first = false;
	}

	return { start_labels, end_labels, container };
}

Value *BytecodeGenerator::visit(const ListComp *node)
{
	const std::string &function_name = Mangler::default_mangler().function_mangle(
		mangle_namespace(m_stack), "<listcomp>", node->source_location());
	auto *f = create_function(function_name);
	create_nested_scope("<listcomp>", function_name);

	std::vector<std::pair<std::string, BytecodeFreeValue *>> captures;
	for (const auto &capture : m_variable_visibility.at(function_name)->captures) {
		auto *value = create_free_value(capture);
		captures.emplace_back(capture, value);
		m_stack.top().locals.emplace(capture, value);
	}

	auto old_function_id = m_function_id;
	m_function_id = f->function_info().function_id;

	auto *block = allocate_block(m_function_id);
	auto *old_block = m_current_block;
	set_insert_point(block);
	auto [start_labels, end_labels, list] =
		visit_comprehension(node->generators(), [this]() { return build_list({}); });
	auto *element = generate(node->elt().get(), m_function_id);
	ASSERT(element)
	emit<ListAppend>(list->get_register(), element->get_register());
	ASSERT(start_labels.size() == end_labels.size());
	while (!start_labels.empty()) {
		auto start_label = start_labels.back();
		auto end_label = end_labels.back();
		emit<Jump>(start_label);
		bind(*end_label);
		start_labels.pop_back();
		end_labels.pop_back();
	}
	emit<ReturnValue>(list->get_register());

	m_stack.pop();
	exit_function(f->function_info().function_id);
	m_function_id = old_function_id;
	set_insert_point(old_block);

	auto captures_tuple = [&]() -> std::optional<Register> {
		if (!captures.empty()) {
			std::vector<Register> capture_regs;
			capture_regs.reserve(captures.size());
			for (const auto &[name, el] : captures) {
				ASSERT(m_stack.top().locals.contains(name));
				const auto &value = m_stack.top().locals.at(name);
				ASSERT(std::holds_alternative<BytecodeFreeValue *>(value))
				emit<LoadClosure>(el->get_free_var_index(),
					std::get<BytecodeFreeValue *>(value)->get_free_var_index());
				capture_regs.push_back(el->get_free_var_index());
			}
			auto *tuple_value = build_tuple(capture_regs);
			return tuple_value->get_register();
		} else {
			return {};
		}
	}();

	std::vector<std::string> varnames{
		"comprehension_iterator",
	};
	const auto &name_visibility_it = m_variable_visibility.find(function_name);
	ASSERT(name_visibility_it != m_variable_visibility.end())
	const auto &name_visibility = name_visibility_it->second->visibility;
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
	f->function_info().function.metadata.nlocals = varnames.size();
	f->function_info().function.metadata.varnames = std::move(varnames);
	f->function_info().function.metadata.arg_count = 1;
	f->function_info().function.metadata.kwonly_arg_count = 0;
	f->function_info().function.metadata.cell2arg = {};
	f->function_info().function.metadata.flags = CodeFlags::create();
	make_function(f->get_register(), f->get_name(), {}, {}, captures_tuple);
	auto *generator = node->generators()[0].get();
	auto *iterable = generate(generator->iter().get(), m_function_id);
	auto iterator = create_value();
	emit<GetIter>(iterator->get_register(), iterable->get_register());
	emit_call(f->get_register(), { iterator->get_register() });
	return create_return_value();
}

Value *BytecodeGenerator::visit(const DictComp *node)
{
	const std::string &function_name = Mangler::default_mangler().function_mangle(
		mangle_namespace(m_stack), "<dictcomp>", node->source_location());
	auto *f = create_function(function_name);
	create_nested_scope("<dictcomp>", function_name);

	std::vector<std::pair<std::string, BytecodeFreeValue *>> captures;
	for (const auto &capture : m_variable_visibility.at(function_name)->captures) {
		auto *value = create_free_value(capture);
		captures.emplace_back(capture, value);
		m_stack.top().locals.emplace(capture, value);
	}

	auto old_function_id = m_function_id;
	m_function_id = f->function_info().function_id;

	auto *block = allocate_block(m_function_id);
	auto *old_block = m_current_block;
	set_insert_point(block);
	auto [start_labels, end_labels, dict] =
		visit_comprehension(node->generators(), [this]() { return build_dict({}, {}); });
	auto *key = generate(node->key().get(), m_function_id);
	ASSERT(key);
	auto *value = generate(node->value().get(), m_function_id);
	ASSERT(value);
	emit<DictAdd>(dict->get_register(), key->get_register(), value->get_register());
	ASSERT(start_labels.size() == end_labels.size());
	while (!start_labels.empty()) {
		auto start_label = start_labels.back();
		auto end_label = end_labels.back();
		emit<Jump>(start_label);
		bind(*end_label);
		start_labels.pop_back();
		end_labels.pop_back();
	}
	emit<ReturnValue>(dict->get_register());

	m_stack.pop();
	exit_function(f->function_info().function_id);
	m_function_id = old_function_id;
	set_insert_point(old_block);

	auto captures_tuple = [&]() -> std::optional<Register> {
		if (!captures.empty()) {
			std::vector<Register> capture_regs;
			capture_regs.reserve(captures.size());
			for (const auto &[name, el] : captures) {
				ASSERT(m_stack.top().locals.contains(name));
				const auto &value = m_stack.top().locals.at(name);
				ASSERT(std::holds_alternative<BytecodeFreeValue *>(value))
				emit<LoadClosure>(el->get_free_var_index(),
					std::get<BytecodeFreeValue *>(value)->get_free_var_index());
				capture_regs.push_back(el->get_free_var_index());
			}
			auto *tuple_value = build_tuple(capture_regs);
			return tuple_value->get_register();
		} else {
			return {};
		}
	}();

	std::vector<std::string> varnames{
		"comprehension_iterator",
	};
	const auto &name_visibility_it = m_variable_visibility.find(function_name);
	ASSERT(name_visibility_it != m_variable_visibility.end())
	const auto &name_visibility = name_visibility_it->second->visibility;
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
	f->function_info().function.metadata.nlocals = varnames.size();
	f->function_info().function.metadata.varnames = std::move(varnames);
	f->function_info().function.metadata.arg_count = 1;
	f->function_info().function.metadata.kwonly_arg_count = 0;
	f->function_info().function.metadata.cell2arg = {};
	f->function_info().function.metadata.flags = CodeFlags::create();
	make_function(f->get_register(), f->get_name(), {}, {}, captures_tuple);
	auto *generator = node->generators()[0].get();
	auto *iterable = generate(generator->iter().get(), m_function_id);
	auto iterator = create_value();
	emit<GetIter>(iterator->get_register(), iterable->get_register());
	emit_call(f->get_register(), { iterator->get_register() });
	return create_return_value();
}

Value *BytecodeGenerator::visit(const GeneratorExp *node)
{
	// FIXME: this should be a generator, ie. produce values lazily
	const std::string &function_name = Mangler::default_mangler().function_mangle(
		mangle_namespace(m_stack), "<genexpr>", node->source_location());
	auto *f = create_function(function_name);
	create_nested_scope("<genexpr>", function_name);

	std::vector<std::pair<std::string, BytecodeFreeValue *>> captures;
	for (const auto &capture : m_variable_visibility.at(function_name)->captures) {
		auto *value = create_free_value(capture);
		captures.emplace_back(capture, value);
		m_stack.top().locals.emplace(capture, value);
	}

	auto old_function_id = m_function_id;
	m_function_id = f->function_info().function_id;

	auto *block = allocate_block(m_function_id);
	auto *old_block = m_current_block;
	set_insert_point(block);
	auto [start_labels, end_labels, list] =
		visit_comprehension(node->generators(), [this]() { return build_list({}); });
	auto *element = generate(node->elt().get(), m_function_id);
	ASSERT(element)
	emit<ListAppend>(list->get_register(), element->get_register());
	ASSERT(start_labels.size() == end_labels.size());
	while (!start_labels.empty()) {
		auto start_label = start_labels.back();
		auto end_label = end_labels.back();
		emit<Jump>(start_label);
		bind(*end_label);
		start_labels.pop_back();
		end_labels.pop_back();
	}
	emit<ReturnValue>(list->get_register());

	m_stack.pop();
	exit_function(f->function_info().function_id);
	m_function_id = old_function_id;
	set_insert_point(old_block);

	auto captures_tuple = [&]() -> std::optional<Register> {
		if (!captures.empty()) {
			std::vector<Register> capture_regs;
			capture_regs.reserve(captures.size());
			for (const auto &[name, el] : captures) {
				ASSERT(m_stack.top().locals.contains(name));
				const auto &value = m_stack.top().locals.at(name);
				ASSERT(std::holds_alternative<BytecodeFreeValue *>(value))
				emit<LoadClosure>(el->get_free_var_index(),
					std::get<BytecodeFreeValue *>(value)->get_free_var_index());
				capture_regs.push_back(el->get_free_var_index());
			}
			auto *tuple_value = build_tuple(capture_regs);
			return tuple_value->get_register();
		} else {
			return {};
		}
	}();

	std::vector<std::string> varnames{
		"comprehension_iterator",
	};
	const auto &name_visibility_it = m_variable_visibility.find(function_name);
	ASSERT(name_visibility_it != m_variable_visibility.end())
	const auto &name_visibility = name_visibility_it->second->visibility;
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
	f->function_info().function.metadata.nlocals = varnames.size();
	f->function_info().function.metadata.varnames = std::move(varnames);
	f->function_info().function.metadata.arg_count = 1;
	f->function_info().function.metadata.kwonly_arg_count = 0;
	f->function_info().function.metadata.cell2arg = {};
	f->function_info().function.metadata.flags = CodeFlags::create();
	make_function(f->get_register(), f->get_name(), {}, {}, captures_tuple);
	auto *generator = node->generators()[0].get();
	auto *iterable = generate(generator->iter().get(), m_function_id);
	auto iterator = create_value();
	emit<GetIter>(iterator->get_register(), iterable->get_register());
	emit_call(f->get_register(), { iterator->get_register() });
	return create_return_value();
}

Value *BytecodeGenerator::visit(const SetComp *node)
{
	const std::string &function_name = Mangler::default_mangler().function_mangle(
		mangle_namespace(m_stack), "<setcomp>", node->source_location());
	auto *f = create_function(function_name);
	create_nested_scope("<setcomp>", function_name);

	std::vector<std::pair<std::string, BytecodeFreeValue *>> captures;
	for (const auto &capture : m_variable_visibility.at(function_name)->captures) {
		auto *value = create_free_value(capture);
		captures.emplace_back(capture, value);
		m_stack.top().locals.emplace(capture, value);
	}

	auto old_function_id = m_function_id;
	m_function_id = f->function_info().function_id;

	auto *block = allocate_block(m_function_id);
	auto *old_block = m_current_block;
	set_insert_point(block);
	auto [start_labels, end_labels, list] =
		visit_comprehension(node->generators(), [this]() { return build_set({}); });
	auto *element = generate(node->elt().get(), m_function_id);
	ASSERT(element)
	emit<SetAdd>(list->get_register(), element->get_register());
	ASSERT(start_labels.size() == end_labels.size());
	while (!start_labels.empty()) {
		auto start_label = start_labels.back();
		auto end_label = end_labels.back();
		emit<Jump>(start_label);
		bind(*end_label);
		start_labels.pop_back();
		end_labels.pop_back();
	}
	emit<ReturnValue>(list->get_register());

	m_stack.pop();
	exit_function(f->function_info().function_id);
	m_function_id = old_function_id;
	set_insert_point(old_block);

	auto captures_tuple = [&]() -> std::optional<Register> {
		if (!captures.empty()) {
			std::vector<Register> capture_regs;
			capture_regs.reserve(captures.size());
			for (const auto &[name, el] : captures) {
				ASSERT(m_stack.top().locals.contains(name));
				const auto &value = m_stack.top().locals.at(name);
				ASSERT(std::holds_alternative<BytecodeFreeValue *>(value))
				emit<LoadClosure>(el->get_free_var_index(),
					std::get<BytecodeFreeValue *>(value)->get_free_var_index());
				capture_regs.push_back(el->get_free_var_index());
			}
			auto *tuple_value = build_tuple(capture_regs);
			return tuple_value->get_register();
		} else {
			return {};
		}
	}();

	std::vector<std::string> varnames{
		"comprehension_iterator",
	};
	const auto &name_visibility_it = m_variable_visibility.find(function_name);
	ASSERT(name_visibility_it != m_variable_visibility.end())
	const auto &name_visibility = name_visibility_it->second->visibility;
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
	f->function_info().function.metadata.nlocals = varnames.size();
	f->function_info().function.metadata.varnames = std::move(varnames);
	f->function_info().function.metadata.arg_count = 1;
	f->function_info().function.metadata.kwonly_arg_count = 0;
	f->function_info().function.metadata.cell2arg = {};
	f->function_info().function.metadata.flags = CodeFlags::create();
	make_function(f->get_register(), f->get_name(), {}, {}, captures_tuple);
	auto *generator = node->generators()[0].get();
	auto *iterable = generate(generator->iter().get(), m_function_id);
	auto iterator = create_value();
	emit<GetIter>(iterator->get_register(), iterable->get_register());
	emit_call(f->get_register(), { iterator->get_register() });
	return create_return_value();
}

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
			for (const auto &ins : block) { ins->relocate(instruction_idx++); }
		}
	}
}

std::shared_ptr<Program> BytecodeGenerator::generate_executable(std::string filename,
	std::vector<std::string> argv)
{
	ASSERT(m_frame_register_count.size() == 2)
	ASSERT(m_frame_stack_value_count.size() == 2)
	ASSERT(m_frame_free_var_count.size() == 2)
	relocate_labels(m_functions);
	return BytecodeProgram::create(std::move(m_functions), filename, argv);
}

InstructionBlock *BytecodeGenerator::allocate_block(size_t function_id)
{
	ASSERT(function_id < m_functions.functions.size())

	auto function = std::next(m_functions.functions.begin(), function_id);
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
	// if (!m_stack.empty()) {
	// 	for (const auto &[k, v] : m_stack.top().locals) {
	// 		if (std::holds_alternative<BytecodeFreeValue *>(v)) { locals[k] = v; }
	// 	}
	// }
	m_stack.push(Scope{ .name = name, .mangled_name = mangled_name, .locals = locals });
}
}// namespace codegen
