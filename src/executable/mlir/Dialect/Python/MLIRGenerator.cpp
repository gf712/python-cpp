#include "Python/MLIRGenerator.hpp"
#include "Python/IR/Dialect.hpp"
#include "Python/IR/PythonOps.hpp"

#include "ast/AST.hpp"
#include "executable/Mangler.hpp"
#include "executable/Program.hpp"
#include "executable/mlir/Conversion/Passes.hpp"
#include "executable/mlir/Conversion/PythonToPythonBytecode/PythonToPythonBytecode.hpp"
#include "executable/mlir/Target/PythonBytecode/PythonBytecodeEmitter.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "runtime/Value.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/RegionUtils.h"
#include "utilities.hpp"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <ranges>
#include <variant>

namespace fs = std::filesystem;

namespace {

mlir::Location loc(mlir::OpBuilder &builder, std::string_view filename, const SourceLocation &loc)
{
	return mlir::FileLineColLoc::get(
		builder.getStringAttr(filename), loc.start.row, loc.start.column);
}

void add_name(mlir::OpBuilder &builder, mlir::StringRef name, mlir::Operation *fn)
{
	ASSERT(mlir::isa<mlir::func::FuncOp>(fn) || mlir::isa<mlir::py::ClassDefinitionOp>(fn));
	if (fn->hasAttr("names")) {
		auto names = fn->getAttr("names");
		std::vector<mlir::StringRef> names_vec;
		auto arr = names.cast<mlir::ArrayAttr>().getValue();
		if (std::find_if(arr.begin(),
				arr.end(),
				[name](mlir::Attribute attr) {
					return attr.cast<mlir::StringAttr>().getValue() == name;
				})
			!= arr.end()) {
			return;
		}
		std::transform(
			arr.begin(), arr.end(), std::back_inserter(names_vec), [](mlir::Attribute attr) {
				return attr.cast<mlir::StringAttr>().getValue();
			});
		names_vec.emplace_back(name);
		fn->setAttr("names", builder.getStrArrayAttr(names_vec));
	} else {
		fn->setAttr("names", builder.getStrArrayAttr({ name }));
	}
}

void add_cell_variable(mlir::OpBuilder &builder, mlir::StringRef name, mlir::Operation *fn)
{
	ASSERT(mlir::isa<mlir::func::FuncOp>(fn) || mlir::isa<mlir::py::ClassDefinitionOp>(fn));

	if (fn->hasAttr("cellvars")) {
		auto names = fn->getAttr("cellvars");
		std::vector<mlir::StringRef> names_vec;
		auto arr = names.cast<mlir::ArrayAttr>().getValue();
		if (std::find_if(arr.begin(),
				arr.end(),
				[name](mlir::Attribute attr) {
					return attr.cast<mlir::StringAttr>().getValue() == name;
				})
			!= arr.end()) {
			return;
		}
		std::transform(
			arr.begin(), arr.end(), std::back_inserter(names_vec), [](mlir::Attribute attr) {
				return attr.cast<mlir::StringAttr>().getValue();
			});
		names_vec.emplace_back(name);
		fn->setAttr("cellvars", builder.getStrArrayAttr(names_vec));
	} else {
		fn->setAttr("cellvars", builder.getStrArrayAttr({ name }));
	}
}

void add_free_variable(mlir::OpBuilder &builder, mlir::StringRef name, mlir::Operation *fn)
{
	ASSERT(mlir::isa<mlir::func::FuncOp>(fn) || mlir::isa<mlir::py::ClassDefinitionOp>(fn));

	if (fn->hasAttr("freevars")) {
		auto names = fn->getAttr("freevars");
		std::vector<mlir::StringRef> names_vec;
		auto arr = names.cast<mlir::ArrayAttr>().getValue();
		if (std::find_if(arr.begin(),
				arr.end(),
				[name](mlir::Attribute attr) {
					return attr.cast<mlir::StringAttr>().getValue() == name;
				})
			!= arr.end()) {
			return;
		}
		std::transform(
			arr.begin(), arr.end(), std::back_inserter(names_vec), [](mlir::Attribute attr) {
				return attr.cast<mlir::StringAttr>().getValue();
			});
		names_vec.emplace_back(name);
		fn->setAttr("freevars", builder.getStrArrayAttr(names_vec));
	} else {
		fn->setAttr("freevars", builder.getStrArrayAttr({ name }));
	}
}

mlir::py::ConstantOp load_const(mlir::OpBuilder &builder,
	py::BigIntType integer,
	std::string_view filename,
	SourceLocation source_location)
{
	auto value = [&]() {
		if (integer == 0) {
			return builder.getIntegerAttr(
				builder.getIntegerType(1, false), llvm::APInt::getZero(1));
		} else {
			const size_t bits = mpz_sizeinbase(integer.get_mpz_t(), 2);
			return builder.getIntegerAttr(
				builder.getIntegerType(bits, integer.get_mpz_t()->_mp_size < 0),
				llvm::APInt(bits,
					llvm::ArrayRef<uint64_t>(
						integer.get_mpz_t()->_mp_d, std::abs(integer.get_mpz_t()->_mp_size))));
		}
	}();

	auto op = builder.create<mlir::py::ConstantOp>(loc(builder, filename, source_location), value);
	return op;
}

mlir::py::ConstantOp load_const(mlir::OpBuilder &builder,
	std::string_view str,
	std::string_view filename,
	SourceLocation source_location)
{
	return builder.create<mlir::py::ConstantOp>(
		loc(builder, filename, source_location), builder.getStringAttr(str));
}

/// Find the first parent operation of the given type, or nullptr if there is
/// no ancestor operation.
template<typename... ParentTs> mlir::Operation *getParentOfType(mlir::Region *region)
{
	do {
		if ((... || mlir::isa<ParentTs>(*region->getParentOp()))) return region->getParentOp();
	} while ((region = region->getParentRegion()));
	return nullptr;
}

}// namespace

namespace codegen {


class SSABuilder
{
	std::unordered_map<std::string, std::map<mlir::Block *, mlir::Value>> m_current_def;

  public:
	void write_variable(std::string varname, mlir::Block *block, mlir::Value value)
	{
		m_current_def[varname][block] = std::move(value);
	}

	mlir::Value read_variable(std::string varname, mlir::Block *block)
	{
		if (auto var_block_it = m_current_def.find(varname); var_block_it != m_current_def.end()) {
			if (auto it = var_block_it->second.find(block); it != var_block_it->second.end()) {
				// local value numbering
				return it->second;
			}
		}
		return read_variable_recursive(std::move(varname), block);
	}

  private:
	mlir::Value read_variable_recursive(std::string varname, mlir::Block *block) { TODO(); }
};

struct Context::ContextImpl
{
	mlir::MLIRContext m_ctx;
	std::unique_ptr<mlir::OpBuilder> m_builder;
	mlir::ModuleOp m_module;

	ContextImpl()
	{
		m_ctx.getOrLoadDialect<mlir::py::PythonDialect>();
		m_builder = std::make_unique<mlir::OpBuilder>(&m_ctx);
		m_module = mlir::ModuleOp::create(m_builder->getUnknownLoc());
	}

	mlir::Type pyobject_type() { return mlir::py::PyObjectType::get(&m_ctx); }

	mlir::py::PyEllipsisType pyellipsis_type() { return mlir::py::PyEllipsisType::get(&m_ctx); }
};

mlir::MLIRContext &Context::ctx() { return m_impl->m_ctx; }

mlir::OpBuilder &Context::builder() { return *m_impl->m_builder; }

mlir::ModuleOp &Context::module() { return m_impl->m_module; }

std::string_view Context::filename() const
{
	return mlir::cast<mlir::FileLineColLoc>(m_impl->m_module.getLoc()).getFilename().strref();
}

Context::Context(std::unique_ptr<ContextImpl> impl) : m_impl(std::move(impl)) {}

Context::~Context() = default;

Context Context::create() { return Context{ std::make_unique<ContextImpl>() }; }

MLIRGenerator::MLIRGenerator(Context &ctx) : m_context(ctx)
{
	m_context.ctx().loadDialect<mlir::cf::ControlFlowDialect>();
	m_context.ctx().loadDialect<mlir::func::FuncDialect>();
	m_context.ctx().loadDialect<mlir::scf::SCFDialect>();
	// m_builder = std::make_unique<SSABuilder>();
}

bool MLIRGenerator::compile(std::shared_ptr<ast::Module> m,
	std::vector<std::string> argv,
	Context &ctx)
{
	MLIRGenerator generator{ ctx };
	generator.m_variable_visibility = VariablesResolver::resolve(m.get());
	m->codegen(&generator);
	std::vector<mlir::StringRef> argv_ref;
	argv_ref.reserve(argv.size());
	for (const auto &argv_ : argv) { argv_ref.push_back(argv_); }
	ctx.module()->setAttr(
		ctx.builder().getStringAttr("llvm.argv"), ctx.builder().getStrArrayAttr(argv_ref));

	// ctx.module().print(llvm::outs());
	// llvm::outs() << '\n';

	// some blocks may be unreachable, which may cause issues when running
	// applyFullConversion where PythonDialect is an illegal target
	// so let's remove them, as no one should rely on the existence of unreachable blocks
	mlir::IRRewriter rewriter{ &ctx.ctx() };
	(void)mlir::eraseUnreachableBlocks(rewriter, { ctx.module().getRegion() });

	if (ctx.module().verify().failed()) {
		ctx.module().print(llvm::outs());
		llvm::outs() << '\n';
		std::abort();
	}

	return true;
}

struct MLIRGenerator::MLIRValue : public ast::Value
{
	mlir::Value value;

	static std::string get_name(const mlir::Value &v)
	{
		if (auto *op = v.getDefiningOp()) { return op->getName().getStringRef().str(); }
		return "";
	}

	MLIRValue(mlir::Value value_) : ast::Value(get_name(value_)), value(std::move(value_)) {}
};

std::optional<mlir::Block *> MLIRGenerator::unhappy_path() const
{
	if (scope().unhappy_path.empty()) { return std::nullopt; }
	return scope().unhappy_path.top();
}

template<typename... Args> MLIRGenerator::MLIRValue *MLIRGenerator::new_value(Args &&...args)
{
	return m_values.emplace_back(std::make_unique<MLIRValue>(std::forward<Args>(args)...)).get();
}


ast::Value *MLIRGenerator::visit(const ast::Argument *node)
{
	TODO();
	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::Arguments *node)
{
	TODO();
	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::Attribute *node)
{
	auto self = static_cast<MLIRValue *>(node->value()->codegen(this))->value;
	switch (node->context()) {
	case ast::ContextType::LOAD: {
		return new_value(m_context.builder().create<mlir::py::LoadAttributeOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			self,
			m_context.builder().getStringAttr(node->attr())));
	} break;
	case ast::ContextType::STORE: {
		TODO();
	} break;
	case ast::ContextType::DELETE: {
		m_context.builder().create<mlir::py::DeleteAttributeOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			self,
			m_context.builder().getStringAttr(node->attr()));
		return nullptr;
	} break;
	case ast::ContextType::UNSET: {
		TODO();
	} break;
	}
	TODO();
	return nullptr;
}

void MLIRGenerator::store_name(std::string_view name,
	MLIRGenerator::MLIRValue *value,
	const SourceLocation &location)
{
	const auto &visibility_map = m_variable_visibility.at(scope().mangled_name);

	const auto visibility = [&] {
		if (auto it = visibility_map->symbol_map.get_hidden_symbol(std::string{ name });
			it.has_value()) {
			ASSERT(visibility_map->type == VariablesResolver::Scope::Type::CLASS);
			return it->get().visibility;
		} else if (auto it = visibility_map->symbol_map.get_visible_symbol(std::string{ name });
				   it.has_value()) {
			return it->get().visibility;
		} else {
			TODO();
		}
	}();

	switch (visibility) {
	case VariablesResolver::Visibility::NAME: {
		m_context.builder().create<mlir::py::StoreNameOp>(
			loc(m_context.builder(), m_context.filename(), location),
			m_context->pyobject_type(),
			name,
			value->value);
	} break;
	case VariablesResolver::Visibility::LOCAL: {
		m_context.builder().create<mlir::py::StoreFastOp>(
			loc(m_context.builder(), m_context.filename(), location),
			m_context->pyobject_type(),
			name,
			value->value);
	} break;
	case VariablesResolver::Visibility::EXPLICIT_GLOBAL:
	case VariablesResolver::Visibility::IMPLICIT_GLOBAL: {
		if (&m_scope.front() == &scope()) {
			m_context.builder().create<mlir::py::StoreNameOp>(
				loc(m_context.builder(), m_context.filename(), location),
				m_context->pyobject_type(),
				name,
				value->value);
		} else {
			auto current_fn = getParentOfType<mlir::func::FuncOp, mlir::py::ClassDefinitionOp>(
				m_context.builder().getInsertionBlock()->getParent());
			add_name(m_context.builder(), name, current_fn);
			m_context.builder().create<mlir::py::StoreGlobalOp>(
				loc(m_context.builder(), m_context.filename(), location),
				m_context->pyobject_type(),
				name,
				value->value);
		}
	} break;
	case VariablesResolver::Visibility::CELL: {
		auto parent = getParentOfType<mlir::func::FuncOp, mlir::py::ClassDefinitionOp>(
			m_context.builder().getInsertionBlock()->getParent());
		auto arr = parent->getAttr("cellvars").cast<mlir::ArrayAttr>().getValue();
		ASSERT(std::find_if(arr.begin(), arr.end(), [name](mlir::Attribute attr) {
			return attr.cast<mlir::StringAttr>().getValue() == mlir::StringRef{ name };
		}) != arr.end());
		m_context.builder().create<mlir::py::StoreDerefOp>(
			loc(m_context.builder(), m_context.filename(), location),
			m_context->pyobject_type(),
			name,
			value->value);
	} break;
	case VariablesResolver::Visibility::FREE: {
		auto parent = getParentOfType<mlir::func::FuncOp, mlir::py::ClassDefinitionOp>(
			m_context.builder().getInsertionBlock()->getParent());
		auto arr = parent->getAttr("freevars").cast<mlir::ArrayAttr>().getValue();
		ASSERT(std::find_if(arr.begin(), arr.end(), [name](mlir::Attribute attr) {
			return attr.cast<mlir::StringAttr>().getValue() == mlir::StringRef{ name };
		}) != arr.end());
		m_context.builder().create<mlir::py::StoreDerefOp>(
			loc(m_context.builder(), m_context.filename(), location),
			m_context->pyobject_type(),
			name,
			value->value);
	} break;
	case VariablesResolver::Visibility::HIDDEN: {
		m_context.builder().create<mlir::py::StoreNameOp>(
			loc(m_context.builder(), m_context.filename(), location),
			m_context->pyobject_type(),
			name,
			value->value);
	} break;
	}
}

MLIRGenerator::MLIRValue *MLIRGenerator::load_name(std::string_view name,
	const SourceLocation &location)
{
	const auto &visibility_map = m_variable_visibility.at(scope().mangled_name);

	const auto visibility = [&] {
		if (auto it = visibility_map->symbol_map.get_hidden_symbol(std::string{ name });
			it.has_value()) {
			ASSERT(visibility_map->type == VariablesResolver::Scope::Type::CLASS);
			return it->get().visibility;
		} else if (auto it = visibility_map->symbol_map.get_visible_symbol(std::string{ name });
				   it.has_value()) {
			return it->get().visibility;
		} else {
			TODO();
		}
	}();

	switch (visibility) {
	case VariablesResolver::Visibility::NAME: {
		return new_value(m_context.builder().create<mlir::py::LoadNameOp>(
			loc(m_context.builder(), m_context.filename(), location),
			m_context->pyobject_type(),
			name));
	}
	case VariablesResolver::Visibility::LOCAL: {
		return new_value(m_context.builder().create<mlir::py::LoadFastOp>(
			loc(m_context.builder(), m_context.filename(), location),
			m_context->pyobject_type(),
			name));
	} break;
	case VariablesResolver::Visibility::EXPLICIT_GLOBAL:
	case VariablesResolver::Visibility::IMPLICIT_GLOBAL: {
		if (&m_scope.front() == &scope()) {
			return new_value(m_context.builder().create<mlir::py::LoadNameOp>(
				loc(m_context.builder(), m_context.filename(), location),
				m_context->pyobject_type(),
				name));
		}
		auto parent = getParentOfType<mlir::func::FuncOp, mlir::py::ClassDefinitionOp>(
			m_context.builder().getInsertionBlock()->getParent());
		add_name(m_context.builder(), name, parent);
		return new_value(m_context.builder().create<mlir::py::LoadGlobalOp>(
			loc(m_context.builder(), m_context.filename(), location),
			m_context->pyobject_type(),
			name));
	} break;
	case VariablesResolver::Visibility::CELL: {
		auto parent = getParentOfType<mlir::func::FuncOp, mlir::py::ClassDefinitionOp>(
			m_context.builder().getInsertionBlock()->getParent());
		auto arr = parent->getAttr("cellvars").cast<mlir::ArrayAttr>().getValue();
		ASSERT(std::find_if(arr.begin(), arr.end(), [name](mlir::Attribute attr) {
			return attr.cast<mlir::StringAttr>().getValue() == mlir::StringRef{ name };
		}) != arr.end());
		return new_value(m_context.builder().create<mlir::py::LoadDerefOp>(
			loc(m_context.builder(), m_context.filename(), location),
			m_context->pyobject_type(),
			name));
	} break;
	case VariablesResolver::Visibility::FREE: {
		auto parent = getParentOfType<mlir::func::FuncOp, mlir::py::ClassDefinitionOp>(
			m_context.builder().getInsertionBlock()->getParent());
		auto arr = parent->getAttr("freevars").cast<mlir::ArrayAttr>().getValue();
		ASSERT(std::find_if(arr.begin(), arr.end(), [name](mlir::Attribute attr) {
			return attr.cast<mlir::StringAttr>().getValue() == mlir::StringRef{ name };
		}) != arr.end());
		return new_value(m_context.builder().create<mlir::py::LoadDerefOp>(
			loc(m_context.builder(), m_context.filename(), location),
			m_context->pyobject_type(),
			name));
	} break;
	case VariablesResolver::Visibility::HIDDEN: {
		return new_value(m_context.builder().create<mlir::py::LoadNameOp>(
			loc(m_context.builder(), m_context.filename(), location),
			m_context->pyobject_type(),
			name));
	} break;
	}

	ASSERT_NOT_REACHED();
	return nullptr;
}

void MLIRGenerator::delete_name(std::string_view name, const SourceLocation &location)
{
	const auto &visibility_map = m_variable_visibility.at(scope().mangled_name);

	const auto visibility = [&] {
		if (auto it = visibility_map->symbol_map.get_hidden_symbol(std::string{ name });
			it.has_value()) {
			ASSERT(visibility_map->type == VariablesResolver::Scope::Type::CLASS);
			return it->get().visibility;
		} else if (auto it = visibility_map->symbol_map.get_visible_symbol(std::string{ name });
				   it.has_value()) {
			return it->get().visibility;
		} else {
			TODO();
		}
	}();

	switch (visibility) {
	case VariablesResolver::Visibility::EXPLICIT_GLOBAL:
	case VariablesResolver::Visibility::IMPLICIT_GLOBAL: {
		auto current_fn = getParentOfType<mlir::func::FuncOp, mlir::py::ClassDefinitionOp>(
			m_context.builder().getInsertionBlock()->getParent());
		add_name(m_context.builder(), name, current_fn);
		if (&m_scope.front() == &scope()) {
			m_context.builder().create<mlir::py::DeleteNameOp>(
				loc(m_context.builder(), m_context.filename(), location), name);
		} else {
			m_context.builder().create<mlir::py::DeleteGlobalOp>(
				loc(m_context.builder(), m_context.filename(), location), name);
		}
	} break;
	case VariablesResolver::Visibility::NAME: {
		auto current_fn = getParentOfType<mlir::func::FuncOp, mlir::py::ClassDefinitionOp>(
			m_context.builder().getInsertionBlock()->getParent());
		add_name(m_context.builder(), name, current_fn);
		m_context.builder().create<mlir::py::DeleteNameOp>(
			loc(m_context.builder(), m_context.filename(), location), name);
	} break;
	case VariablesResolver::Visibility::LOCAL: {
		m_context.builder().create<mlir::py::DeleteFastOp>(
			loc(m_context.builder(), m_context.filename(), location), name);
	} break;
	case VariablesResolver::Visibility::CELL:
	case VariablesResolver::Visibility::FREE: {
		m_context.builder().create<mlir::py::DeleteDerefOp>(
			loc(m_context.builder(), m_context.filename(), location), name);
	} break;
	case VariablesResolver::Visibility::HIDDEN: {
		auto current_fn = getParentOfType<mlir::func::FuncOp, mlir::py::ClassDefinitionOp>(
			m_context.builder().getInsertionBlock()->getParent());
		add_name(m_context.builder(), name, current_fn);
		m_context.builder().create<mlir::py::DeleteNameOp>(
			loc(m_context.builder(), m_context.filename(), location), name);
	} break;
	}
}

void MLIRGenerator::assign(const std::shared_ptr<ast::ASTNode> &target,
	MLIRValue *src,
	const SourceLocation &source_location)
{
	if (auto ast_name = as<ast::Name>(target)) {
		for (const auto &var : ast_name->ids()) { store_name(var, src, source_location); }
	} else if (auto subscript = as<ast::Subscript>(target)) {
		auto value = static_cast<const MLIRValue &>(*subscript->value()->codegen(this)).value;
		auto index = build_slice(subscript->slice(), subscript->source_location())->value;
		m_context.builder().create<mlir::py::StoreSubscriptOp>(
			loc(m_context.builder(), m_context.filename(), source_location),
			value,
			index,
			src->value);
	} else if (auto tuple = as<ast::Tuple>(target)) {

		if (std::ranges::any_of(tuple->elements(),
				[](const auto &el) -> bool { return as<ast::Starred>(el) != nullptr; })) {
			if (auto starred = as<ast::Starred>(tuple->elements().back())) {
				ASSERT(as<ast::Name>(starred->value()));
				ASSERT(as<ast::Name>(starred->value())->context_type() == ast::ContextType::STORE);
				std::vector<mlir::Value> unpacked_values;
				std::vector<mlir::Type> unpacked_types(
					tuple->elements().size() - 1, m_context->pyobject_type());
				mlir::Type rest{ m_context->pyobject_type() };
				auto unpack_sequence = m_context.builder().create<mlir::py::UnpackExpandOp>(
					loc(m_context.builder(), m_context.filename(), source_location),
					unpacked_types,
					rest,
					src->value);
				for (const auto &[el, unpacked_value] :
					llvm::zip(tuple->elements(), unpack_sequence.getUnpackedValues())) {
					assign(el, new_value(unpacked_value), source_location);
				}
				assign(as<ast::Name>(starred->value()),
					new_value(unpack_sequence.getRest()),
					source_location);
			} else {
				TODO();
			}
		} else {
			std::vector<mlir::Value> unpacked_values;
			std::vector<mlir::Type> unpacked_types(
				tuple->elements().size(), m_context->pyobject_type());
			auto unpack_sequence = m_context.builder().create<mlir::py::UnpackSequenceOp>(
				loc(m_context.builder(), m_context.filename(), source_location),
				unpacked_types,
				src->value);
			for (const auto &[el, unpacked_value] :
				llvm::zip(tuple->elements(), unpack_sequence.getUnpackedValues())) {
				assign(el, new_value(unpacked_value), source_location);
			}
		}
	} else if (auto attr = as<ast::Attribute>(target)) {
		auto obj = static_cast<const MLIRValue &>(*attr->value()->codegen(this)).value;
		const auto &name = attr->attr();
		m_context.builder().create<mlir::py::StoreAttributeOp>(
			loc(m_context.builder(), m_context.filename(), source_location), obj, name, src->value);
	} else {
		ASSERT(false && "Invalid assignment in AST");
	}
}

ast::Value *MLIRGenerator::visit(const ast::Assign *node)
{
	auto *src = static_cast<MLIRValue *>(node->value()->codegen(this));

	for (const auto &target : node->targets()) { assign(target, src, node->source_location()); }

	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::Assert *node)
{
	auto test = static_cast<const MLIRValue &>(*node->test()->codegen(this)).value;
	auto cond = m_context.builder().create<mlir::py::CastToBoolOp>(
		loc(m_context.builder(), m_context.filename(), node->test()->source_location()),
		m_context.builder().getI1Type(),
		test);

	auto *parent = cond.getOperation()->getParentRegion();
	auto assertion_block = m_context.builder().createBlock(parent);
	auto continuation = m_context.builder().createBlock(parent);

	m_context.builder().setInsertionPointToEnd(cond.getOperation()->getBlock());
	m_context.builder().create<mlir::cf::CondBranchOp>(
		loc(m_context.builder(), m_context.filename(), node->test()->source_location()),
		cond,
		continuation,
		assertion_block);

	m_context.builder().setInsertionPointToEnd(assertion_block);

	auto assertion_error = [this, node]() {
		auto assertion_error_fn = m_context.builder().create<mlir::py::LoadAssertionError>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type());
		if (node->msg()) {
			auto msg = static_cast<const MLIRValue &>(*node->msg()->codegen(this)).value;
			return m_context.builder().create<mlir::py::FunctionCallOp>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				assertion_error_fn,
				std::vector{ msg },
				mlir::DenseStringElementsAttr::get(
					mlir::VectorType::get({ 0 }, mlir::StringAttr::get(&m_context.ctx()).getType()),
					{}),
				std::vector<mlir::Value>{},
				false,
				false);
		} else {
			return m_context.builder().create<mlir::py::FunctionCallOp>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				assertion_error_fn,
				std::vector<mlir::Value>{},
				mlir::DenseStringElementsAttr::get(
					mlir::VectorType::get({ 0 }, mlir::StringAttr::get(&m_context.ctx()).getType()),
					{}),
				std::vector<mlir::Value>{},
				false,
				false);
		}
	}();

	m_context.builder().create<mlir::py::RaiseOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()), assertion_error);

	m_context.builder().setInsertionPointToEnd(continuation);

	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::AsyncFunctionDefinition *node)
{
	const std::string &mangled_name = Mangler::default_mangler().function_mangle(
		mangle_namespace(m_scope), node->name(), node->source_location());
	make_function(node->name(),
		mangled_name,
		node->args(),
		node->body(),
		node->decorator_list(),
		false,
		true,
		node->source_location());
	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::Await *node)
{
	auto iterable = static_cast<const MLIRValue &>(*node->value()->codegen(this)).value;
	auto iterator = m_context.builder().create<mlir::py::GetAwaitableOp>(
		loc(m_context.builder(), m_context.filename(), node->value()->source_location()),
		m_context->pyobject_type(),
		iterable);
	return new_value(m_context.builder().create<mlir::py::YieldFromOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()),
		m_context->pyobject_type(),
		iterator));
}

ast::Value *MLIRGenerator::visit(const ast::AugAssign *node)
{
	std::optional<mlir::Value> target_value;
	std::optional<mlir::Value> target_slice;
	auto *target = [this, node, &target_value, &target_slice]() {
		if (auto named_target = as<ast::Name>(node->target())) {
			ASSERT(named_target->context_type() == ast::ContextType::STORE);
			if (named_target->ids().size() != 1) { TODO(); }
			return load_name(named_target->ids()[0], node->target()->source_location());
		} else if (auto attribute_target = as<ast::Attribute>(node->target())) {
			target_value =
				static_cast<MLIRValue &>(*attribute_target->value()->codegen(this)).value;
			return new_value(m_context.builder().create<mlir::py::LoadAttributeOp>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				*target_value,
				m_context.builder().getStringAttr(attribute_target->attr())));
		} else if (auto subscript_target = as<ast::Subscript>(node->target())) {
			target_value =
				static_cast<MLIRValue &>(*subscript_target->value()->codegen(this)).value;
			target_slice =
				build_slice(subscript_target->slice(), subscript_target->source_location())->value;
			return new_value(m_context.builder().create<mlir::py::BinarySubscriptOp>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				*target_value,
				*target_slice));
		} else {
			std::cerr << fmt::format("Invalid node type ({}) for augmented assignment",
				ast::node_type_to_string(node->node_type()));
			ASSERT_NOT_REACHED();
		}
	}();
	auto value = node->value()->codegen(this);

	auto result = [&]() {
		auto make_binop = [this, &node](
							  ast::Value *value, ast::Value *target, mlir::py::InplaceOpKind kind) {
			return new_value(m_context.builder().create<mlir::py::InplaceOp>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				static_cast<MLIRValue *>(value)->value,
				static_cast<MLIRValue *>(target)->value,
				mlir::py::InplaceOpKindAttr::get(&m_context.ctx(), kind)));
		};
		switch (node->op()) {
		case ast::BinaryOpType::PLUS: {
			return make_binop(value, target, mlir::py::InplaceOpKind::add);
		} break;
		case ast::BinaryOpType::MINUS: {
			return make_binop(value, target, mlir::py::InplaceOpKind::sub);
		} break;
		case ast::BinaryOpType::MODULO: {
			return make_binop(value, target, mlir::py::InplaceOpKind::mod);
		} break;
		case ast::BinaryOpType::MULTIPLY: {
			return make_binop(value, target, mlir::py::InplaceOpKind::mul);
		} break;
		case ast::BinaryOpType::EXP: {
			return make_binop(value, target, mlir::py::InplaceOpKind::exp);
		} break;
		case ast::BinaryOpType::SLASH: {
			return make_binop(value, target, mlir::py::InplaceOpKind::div);
		} break;
		case ast::BinaryOpType::FLOORDIV: {
			return make_binop(value, target, mlir::py::InplaceOpKind::fldiv);
		} break;
		case ast::BinaryOpType::MATMUL: {
			return make_binop(value, target, mlir::py::InplaceOpKind::mmul);
		} break;
		case ast::BinaryOpType::LEFTSHIFT: {
			return make_binop(value, target, mlir::py::InplaceOpKind::lshift);
		} break;
		case ast::BinaryOpType::RIGHTSHIFT: {
			return make_binop(value, target, mlir::py::InplaceOpKind::rshift);
		} break;
		case ast::BinaryOpType::AND: {
			return make_binop(value, target, mlir::py::InplaceOpKind::and_);
		} break;
		case ast::BinaryOpType::OR: {
			return make_binop(value, target, mlir::py::InplaceOpKind::or_);
		} break;
		case ast::BinaryOpType::XOR: {
			return make_binop(value, target, mlir::py::InplaceOpKind::xor_);
		} break;
		}
		ASSERT_NOT_REACHED();
	}();

	// TODO: fix aug assignment bytecode instruction and store result instead of target
	if (auto named_target = as<ast::Name>(node->target())) {
		if (named_target->ids().size() != 1) { TODO(); }
		store_name(named_target->ids()[0], target, node->target()->source_location());
	} else if (auto attribute_target = as<ast::Attribute>(node->target())) {
		ASSERT(target_value.has_value());
		const auto &name = attribute_target->attr();
		m_context.builder().create<mlir::py::StoreAttributeOp>(
			loc(m_context.builder(), m_context.filename(), node->target()->source_location()),
			*target_value,
			name,
			target->value);
	} else if (as<ast::Subscript>(node->target())) {
		ASSERT(target_value.has_value());
		ASSERT(target_slice.has_value());
		m_context.builder().create<mlir::py::StoreSubscriptOp>(
			loc(m_context.builder(), m_context.filename(), node->target()->source_location()),
			*target_value,
			*target_slice,
			target->value);
	} else {
		ASSERT_NOT_REACHED();
	}

	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::Break *node)
{
	auto *old_b = m_context.builder().getBlock();
	auto *b = m_context.builder().createBlock(old_b->getParent());
	m_context.builder().setInsertionPointToEnd(old_b);
	m_context.builder().create<mlir::cf::BranchOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()), b);
	m_context.builder().setInsertionPointToStart(b);
	m_context.builder().create<mlir::py::ControlFlowYield>(
		loc(m_context.builder(), m_context.filename(), node->source_location()),
		mlir::py::LoopOpKindAttr::get(&m_context.ctx(), mlir::py::LoopOpKind::break_));
	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::BinaryExpr *node)
{
	auto lhs = static_cast<MLIRValue *>(node->lhs()->codegen(this))->value;
	auto rhs = static_cast<MLIRValue *>(node->rhs()->codegen(this))->value;

	switch (node->op_type()) {
	case ast::BinaryOpType::PLUS: {
		auto result = m_context.builder().create<mlir::py::BinaryAddOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			lhs,
			rhs);
		return new_value(result);
	} break;
	case ast::BinaryOpType::MINUS: {
		auto result = m_context.builder().create<mlir::py::BinarySubtractOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			lhs,
			rhs);
		return new_value(result);
	} break;
	case ast::BinaryOpType::MODULO: {
		auto result = m_context.builder().create<mlir::py::BinaryModuloOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			lhs,
			rhs);
		return new_value(result);
	} break;
	case ast::BinaryOpType::MULTIPLY: {
		auto result = m_context.builder().create<mlir::py::BinaryMultiplyOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			lhs,
			rhs);
		return new_value(result);
	} break;
	case ast::BinaryOpType::EXP: {
		auto result = m_context.builder().create<mlir::py::BinaryExpOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			lhs,
			rhs);
		return new_value(result);
	} break;
	case ast::BinaryOpType::SLASH: {
		auto result = m_context.builder().create<mlir::py::BinaryDivOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			lhs,
			rhs);
		return new_value(result);
	} break;
	case ast::BinaryOpType::FLOORDIV: {
		auto result = m_context.builder().create<mlir::py::BinaryFloorDivOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			lhs,
			rhs);
		return new_value(result);
	} break;
	case ast::BinaryOpType::MATMUL: {
		auto result = m_context.builder().create<mlir::py::BinaryMatMulOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			lhs,
			rhs);
		return new_value(result);
	} break;
	case ast::BinaryOpType::LEFTSHIFT: {
		auto result = m_context.builder().create<mlir::py::LeftShiftOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			lhs,
			rhs);
		return new_value(result);
	} break;
	case ast::BinaryOpType::RIGHTSHIFT: {
		auto result = m_context.builder().create<mlir::py::RightShiftOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			lhs,
			rhs);
		return new_value(result);
	} break;
	case ast::BinaryOpType::AND: {
		auto result = m_context.builder().create<mlir::py::LogicalAndOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			lhs,
			rhs);
		return new_value(result);
	} break;
	case ast::BinaryOpType::OR: {
		auto result = m_context.builder().create<mlir::py::LogicalOrOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			lhs,
			rhs);
		return new_value(result);
	} break;
	case ast::BinaryOpType::XOR: {
		auto result = m_context.builder().create<mlir::py::LogicalXorOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			lhs,
			rhs);
		return new_value(result);
	} break;
	}

	ASSERT_NOT_REACHED();
}

ast::Value *MLIRGenerator::visit(const ast::BoolOp *node)
{
	auto *current = m_context.builder().getBlock();
	auto *parent = current->getParent();
	auto *continuation = m_context.builder().createBlock(parent);
	continuation->addArgument(m_context->pyobject_type(), m_context.builder().getUnknownLoc());

	m_context.builder().setInsertionPointToEnd(current);

	switch (node->op()) {
	case ast::BoolOp::OpType::And: {
		auto it = node->values().begin();
		auto end = node->values().end();
		while (std::next(it) != end) {
			auto *result_block = m_context.builder().createBlock(continuation);
			m_context.builder().setInsertionPointToEnd(current);
			m_context.builder().create<mlir::cf::BranchOp>(
				loc(m_context.builder(), m_context.filename(), (*it)->source_location()),
				result_block);
			m_context.builder().setInsertionPointToStart(result_block);
			auto result = static_cast<MLIRValue *>((*it)->codegen(this))->value;
			auto cond = m_context.builder().create<mlir::py::CastToBoolOp>(
				loc(m_context.builder(), m_context.filename(), (*it)->source_location()),
				m_context.builder().getI1Type(),
				result);
			auto *this_block = m_context.builder().getInsertionBlock();
			auto *next = m_context.builder().createBlock(continuation);
			m_context.builder().setInsertionPointToEnd(this_block);
			m_context.builder().create<mlir::cf::CondBranchOp>(
				loc(m_context.builder(), m_context.filename(), (*it)->source_location()),
				cond,
				next,
				mlir::ValueRange{},
				continuation,
				mlir::ValueRange{ result });
			it++;
			current = next;
			m_context.builder().setInsertionPointToEnd(current);
		}
		auto result = static_cast<MLIRValue *>((*it)->codegen(this))->value;
		m_context.builder().create<mlir::cf::BranchOp>(
			loc(m_context.builder(), m_context.filename(), (*it)->source_location()),
			continuation,
			mlir::ValueRange{ result });
		m_context.builder().setInsertionPointToEnd(continuation);
	} break;
	case ast::BoolOp::OpType::Or: {
		auto it = node->values().begin();
		auto end = node->values().end();
		while (std::next(it) != end) {
			auto *result_block = m_context.builder().createBlock(continuation);
			m_context.builder().setInsertionPointToEnd(current);
			m_context.builder().create<mlir::cf::BranchOp>(
				loc(m_context.builder(), m_context.filename(), (*it)->source_location()),
				result_block);
			m_context.builder().setInsertionPointToStart(result_block);
			auto result = static_cast<MLIRValue *>((*it)->codegen(this))->value;
			auto cond = m_context.builder().create<mlir::py::CastToBoolOp>(
				loc(m_context.builder(), m_context.filename(), (*it)->source_location()),
				m_context.builder().getI1Type(),
				result);
			auto *this_block = m_context.builder().getInsertionBlock();
			auto *next = m_context.builder().createBlock(continuation);
			m_context.builder().setInsertionPointToEnd(this_block);
			m_context.builder().create<mlir::cf::CondBranchOp>(
				loc(m_context.builder(), m_context.filename(), (*it)->source_location()),
				cond,
				continuation,
				mlir::ValueRange{ result },
				next,
				mlir::ValueRange{});
			it++;
			current = next;
			m_context.builder().setInsertionPointToEnd(current);
		}
		auto result = static_cast<MLIRValue *>((*it)->codegen(this))->value;
		m_context.builder().create<mlir::cf::BranchOp>(
			loc(m_context.builder(), m_context.filename(), (*it)->source_location()),
			continuation,
			mlir::ValueRange{ result });
		m_context.builder().setInsertionPointToEnd(continuation);
	}
	}

	return new_value(m_context.builder().getBlock()->getArgument(0));
}

ast::Value *MLIRGenerator::visit(const ast::Call *node)
{
	auto callee = [this, node]() -> mlir::Value {
		if (auto method = as<ast::Attribute>(node->function())) {
			auto self = static_cast<MLIRValue *>(method->value()->codegen(this))->value;
			auto method_name = method->attr();
			return m_context.builder().create<mlir::py::LoadMethodOp>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				self,
				method_name);
		} else {
			return static_cast<MLIRValue *>(node->function()->codegen(this))->value;
		}
	}();

	auto is_args_expansion = [](const std::shared_ptr<ast::ASTNode> &node) {
		return node->node_type() == ast::ASTNodeType::Starred;
	};

	auto is_kwargs_expansion = [](const std::shared_ptr<ast::Keyword> &node) {
		return !node->arg().has_value();
	};

	bool requires_args_expansion =
		std::any_of(node->args().begin(), node->args().end(), is_args_expansion);
	bool requires_kwargs_expansion =
		std::any_of(node->keywords().begin(), node->keywords().end(), is_kwargs_expansion);

	std::vector<mlir::Value> arg_values;
	std::vector<mlir::Value> keyword_values;
	std::vector<mlir::StringRef> keywords;

	if (requires_args_expansion || requires_kwargs_expansion) {
		if (!node->args().empty()) { requires_args_expansion = true; }
		{
			std::vector<MLIRValue *> args;
			std::vector<bool> arg_requires_expansion;
			args.reserve(node->args().size());
			arg_requires_expansion.reserve(node->args().size());
			for (const auto &arg : node->args()) {
				auto arg_value = static_cast<MLIRValue *>(arg->codegen(this));
				arg_requires_expansion.push_back(is_args_expansion(arg));
				args.push_back(arg_value);
			}
			arg_values.push_back(static_cast<MLIRValue *>(
				build_tuple(args, arg_requires_expansion, node->source_location()))
									 ->value);
		}
		if (!node->keywords().empty()) { requires_kwargs_expansion = true; }
		{
			std::vector<MLIRValue *> keys;
			std::vector<MLIRValue *> values;
			std::vector<bool> kwarg_requires_expansion;
			keys.reserve(node->keywords().size());
			values.reserve(node->keywords().size());
			kwarg_requires_expansion.reserve(node->keywords().size());

			auto none = new_value(m_context.builder().create<mlir::py::ConstantOp>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context.builder().getNoneType()));
			for (const auto &kwarg : node->keywords()) {
				auto kwarg_value = static_cast<MLIRValue *>(kwarg->codegen(this));
				values.push_back(kwarg_value);
				if (kwarg_requires_expansion.emplace_back(is_kwargs_expansion(kwarg))) {
					keys.push_back(none);
				} else {
					auto name = *kwarg->arg();
					keys.push_back(new_value(m_context.builder().create<mlir::py::ConstantOp>(
						loc(m_context.builder(), m_context.filename(), kwarg->source_location()),
						m_context.builder().getStringAttr(name))));
				}
			}
			keyword_values.push_back(static_cast<MLIRValue *>(
				build_dict(keys, values, kwarg_requires_expansion, node->source_location()))
										 ->value);
		}
	} else {
		arg_values.reserve(node->args().size());
		for (const auto &arg : node->args()) {
			auto *generated_arg = static_cast<MLIRValue *>(arg->codegen(this));
			ASSERT(generated_arg);
			arg_values.push_back(generated_arg->value);
		}

		keyword_values.reserve(node->keywords().size());
		keywords.reserve(node->keywords().size());
		for (const auto &keyword : node->keywords()) {
			keyword_values.push_back(static_cast<MLIRValue *>(keyword->codegen(this))->value);
			const auto &keyword_argname = keyword->arg();
			ASSERT(keyword_argname.has_value());
			keywords.emplace_back(*keyword_argname);
		}
	}

	auto function_call = m_context.builder().create<mlir::py::FunctionCallOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()),
		m_context->pyobject_type(),
		callee,
		arg_values,
		mlir::DenseStringElementsAttr::get(
			mlir::VectorType::get({ static_cast<int64_t>(keywords.size()) },
				mlir::StringAttr::get(&m_context.ctx()).getType()),
			keywords),
		keyword_values,
		requires_args_expansion,
		requires_kwargs_expansion);

	return new_value(function_call);
}

ast::Value *MLIRGenerator::visit(const ast::ClassDefinition *node)
{
	auto *current_block = m_context.builder().getInsertionBlock();
	std::vector<mlir::Value> decorator_functions;
	decorator_functions.reserve(node->decorator_list().size());
	for (const auto &decorator_function : node->decorator_list()) {
		auto *f = decorator_function->codegen(this);
		ASSERT(f);
		decorator_functions.push_back(static_cast<MLIRValue *>(f)->value);
	}

	auto class_mangled_name = Mangler::default_mangler().class_mangle(
		mangle_namespace(m_scope), node->name(), node->source_location());

	const auto &name_visibility_it = m_variable_visibility.find(class_mangled_name);
	ASSERT(name_visibility_it != m_variable_visibility.end());
	const auto &class_scope = name_visibility_it->second;

	std::vector<mlir::Value> bases;
	bases.reserve(node->bases().size());
	for (const auto &base : node->bases()) {
		bases.push_back(static_cast<MLIRValue *>(base->codegen(this))->value);
	}

	std::vector<mlir::StringRef> keywords;
	keywords.reserve(node->keywords().size());
	std::vector<mlir::Value> kwargs;
	kwargs.reserve(node->keywords().size());
	for (const auto &keyword : node->keywords()) {
		ASSERT(keyword->arg().has_value());
		keywords.push_back(*keyword->arg());
		kwargs.push_back(static_cast<MLIRValue *>(keyword->codegen(this))->value);
	}

	auto output = m_context.builder().create<mlir::py::ClassDefinitionOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()),
		m_context->pyobject_type(),
		node->name(),
		class_mangled_name,
		bases,
		mlir::DenseStringElementsAttr::get(
			mlir::VectorType::get({ static_cast<int64_t>(keywords.size()) },
				mlir::StringAttr::get(&m_context.ctx()).getType()),
			keywords),
		kwargs,
		nullptr);

	std::vector<std::string> captures;
	{
		[[maybe_unused]] auto function_scope =
			create_nested_scope(node->name(), class_mangled_name);

		auto *entry_block = &output.getBody().emplaceBlock();
		auto *return_block = &output.getBody().emplaceBlock();

		for (const auto &el : m_variable_visibility.at(class_mangled_name)->symbol_map.symbols) {
			if (el.visibility == VariablesResolver::Visibility::CELL
				&& m_variable_visibility.at(class_mangled_name)->captures.contains(el.name)) {
				captures.push_back(el.name);
			}
		}

		for (const auto &el : m_variable_visibility.at(class_mangled_name)->symbol_map.symbols) {
			if (el.visibility == VariablesResolver::Visibility::FREE
				&& m_variable_visibility.at(class_mangled_name)->captures.contains(el.name)) {
				captures.push_back(el.name);
			}
		}

		std::vector<mlir::StringRef> cellvars;
		std::vector<mlir::StringRef> freevars;
		for (const auto &el : m_variable_visibility.at(class_mangled_name)->symbol_map.symbols) {
			if (el.visibility == VariablesResolver::Visibility::CELL) {
				cellvars.push_back(el.name);
			}
			if (el.visibility == VariablesResolver::Visibility::FREE) {
				freevars.push_back(el.name);
			}
		}
		output->setAttr("cellvars", m_context.builder().getStrArrayAttr(cellvars));
		output->setAttr("freevars", m_context.builder().getStrArrayAttr(freevars));

		m_context.builder().setInsertionPointToStart(entry_block);

		store_name(
			"__module__", load_name("__name__", node->source_location()), node->source_location());
		store_name("__qualname__",
			new_value(load_const(
				m_context.builder(), node->name(), m_context.filename(), node->source_location())),
			node->source_location());

		for (const auto &el : node->body()) { el->codegen(this); }

		m_context.builder().create<mlir::cf::BranchOp>(
			loc(m_context.builder(), m_context.filename(), node->body().back()->source_location()),
			return_block);

		m_context.builder().setInsertionPointToStart(return_block);
		if (class_scope->requires_class_ref) {
			auto *__class__ = load_name("__class__", node->source_location());
			store_name("__classcell__", __class__, node->source_location());
			m_context.builder().create<mlir::func::ReturnOp>(
				m_context.builder().getUnknownLoc(), mlir::ValueRange{ __class__->value });
		} else {
			auto result = m_context.builder().create<mlir::py::ConstantOp>(
				m_context.builder().getUnknownLoc(), m_context.builder().getNoneType());
			m_context.builder().create<mlir::func::ReturnOp>(
				m_context.builder().getUnknownLoc(), mlir::ValueRange{ result });
		}
	}

	m_context.builder().setInsertionPointToEnd(current_block);

	std::vector<mlir::StringRef> captures_ref;
	captures_ref.reserve(captures.size());
	for (const auto &el : captures) { captures_ref.push_back(el); }
	output.setCapturesAttr(mlir::DenseStringElementsAttr::get(
		mlir::VectorType::get({ static_cast<int64_t>(captures.size()) },
			mlir::StringAttr::get(&m_context.ctx()).getType()),
		captures_ref));

	store_name(node->name(), new_value(output), node->source_location());

	if (!decorator_functions.empty()) {
		mlir::Value arg = load_name(node->name(), node->source_location())->value;
		for (const auto &decorator_function : decorator_functions | std::ranges::views::reverse) {
			arg = m_context.builder().create<mlir::py::FunctionCallOp>(decorator_function.getLoc(),
				m_context->pyobject_type(),
				decorator_function,
				mlir::ValueRange{ arg },
				mlir::DenseStringElementsAttr::get(
					mlir::VectorType::get({ 0 }, mlir::StringAttr::get(&m_context.ctx()).getType()),
					{}),
				mlir::ValueRange{},
				false,
				false);
		}
		store_name(node->name(), new_value(arg), node->source_location());
	}

	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::Continue *node)
{
	auto *old_b = m_context.builder().getBlock();
	auto *b = m_context.builder().createBlock(old_b->getParent());
	m_context.builder().setInsertionPointToEnd(old_b);
	m_context.builder().create<mlir::cf::BranchOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()), b);
	m_context.builder().setInsertionPointToStart(b);
	m_context.builder().create<mlir::py::ControlFlowYield>(
		loc(m_context.builder(), m_context.filename(), node->source_location()),
		mlir::py::LoopOpKindAttr::get(&m_context.ctx(), mlir::py::LoopOpKind::continue_));
	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::Compare *node)
{
	std::optional<mlir::py::Compare> result;
	auto lhs = static_cast<MLIRValue *>(node->lhs()->codegen(this))->value;
	const auto &comparators = node->comparators();
	const auto &ops = node->ops();

	for (size_t idx = 0; idx < comparators.size(); ++idx) {
		auto rhs = static_cast<MLIRValue *>(comparators[idx]->codegen(this))->value;
		const auto op = ops[idx];

		switch (op) {
		case ast::Compare::OpType::Eq: {
			result = m_context.builder().create<mlir::py::Compare>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				mlir::py::CmpPredicateAttr::get(&m_context.ctx(), mlir::py::CmpPredicate::eq),
				lhs,
				rhs);
		} break;
		case ast::Compare::OpType::NotEq: {
			result = m_context.builder().create<mlir::py::Compare>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				mlir::py::CmpPredicateAttr::get(&m_context.ctx(), mlir::py::CmpPredicate::ne),
				lhs,
				rhs);
		} break;
		case ast::Compare::OpType::Lt: {
			result = m_context.builder().create<mlir::py::Compare>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				mlir::py::CmpPredicateAttr::get(&m_context.ctx(), mlir::py::CmpPredicate::lt),
				lhs,
				rhs);
		} break;
		case ast::Compare::OpType::LtE: {
			result = m_context.builder().create<mlir::py::Compare>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				mlir::py::CmpPredicateAttr::get(&m_context.ctx(), mlir::py::CmpPredicate::le),
				lhs,
				rhs);
		} break;
		case ast::Compare::OpType::Gt: {
			result = m_context.builder().create<mlir::py::Compare>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				mlir::py::CmpPredicateAttr::get(&m_context.ctx(), mlir::py::CmpPredicate::gt),
				lhs,
				rhs);
		} break;
		case ast::Compare::OpType::GtE: {
			result = m_context.builder().create<mlir::py::Compare>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				mlir::py::CmpPredicateAttr::get(&m_context.ctx(), mlir::py::CmpPredicate::ge),
				lhs,
				rhs);
		} break;
		case ast::Compare::OpType::Is: {
			result = m_context.builder().create<mlir::py::Compare>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				mlir::py::CmpPredicateAttr::get(&m_context.ctx(), mlir::py::CmpPredicate::is),
				lhs,
				rhs);
		} break;
		case ast::Compare::OpType::IsNot: {
			result = m_context.builder().create<mlir::py::Compare>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				mlir::py::CmpPredicateAttr::get(&m_context.ctx(), mlir::py::CmpPredicate::isnot),
				lhs,
				rhs);
		} break;
		case ast::Compare::OpType::In: {
			result = m_context.builder().create<mlir::py::Compare>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				mlir::py::CmpPredicateAttr::get(&m_context.ctx(), mlir::py::CmpPredicate::in),
				lhs,
				rhs);
		} break;
		case ast::Compare::OpType::NotIn: {
			result = m_context.builder().create<mlir::py::Compare>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				mlir::py::CmpPredicateAttr::get(&m_context.ctx(), mlir::py::CmpPredicate::notin),
				lhs,
				rhs);
		} break;
		}
		lhs = rhs;
	}

	ASSERT(result.has_value());
	return new_value(*result);
}

ast::Value *MLIRGenerator::visit(const ast::Comprehension *node)
{
	ASSERT_NOT_REACHED();
	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::Constant *node)
{
	const auto &value = *node->value();
	return std::visit(
		overloaded{
			[this, node](const py::Number &number) -> ast::Value * {
				return std::visit(overloaded{
									  [this, node](double value) {
										  mlir::py::ConstantOp op =
											  m_context.builder().create<mlir::py::ConstantOp>(
												  loc(m_context.builder(),
													  m_context.filename(),
													  node->source_location()),
												  value);
										  return new_value(op);
									  },
									  [this, node](const py::BigIntType &int_value) {
										  return new_value(load_const(m_context.builder(),
											  int_value,
											  m_context.filename(),
											  node->source_location()));
									  },
								  },
					number.value);
			},
			[this, node](const py::NameConstant &c) -> ast::Value * {
				return std::visit(
					overloaded{
						[this, node](bool value) {
							auto op = m_context.builder().create<mlir::py::ConstantOp>(
								loc(m_context.builder(),
									m_context.filename(),
									node->source_location()),
								value);
							return new_value(op);
						},
						[this, node](py::NoneType) {
							auto op = m_context.builder().create<mlir::py::ConstantOp>(
								loc(m_context.builder(),
									m_context.filename(),
									node->source_location()),
								m_context.builder().getNoneType());
							return new_value(op);
						},
					},
					c.value);
			},
			[this, node](const py::String &s) -> ast::Value * {
				return new_value(load_const(
					m_context.builder(), s.s, m_context.filename(), node->source_location()));
			},
			[this, node](const py::Bytes &b) -> ast::Value * {
				mlir::py::ConstantOp op = m_context.builder().create<mlir::py::ConstantOp>(
					loc(m_context.builder(), m_context.filename(), node->source_location()), b.b);
				return new_value(op);
			},
			[this, node](py::Ellipsis) -> ast::Value * {
				mlir::py::ConstantOp op = m_context.builder().create<mlir::py::ConstantOp>(
					loc(m_context.builder(), m_context.filename(), node->source_location()),
					m_context->pyellipsis_type());
				return new_value(op);
			},
			[](auto) -> ast::Value * {
				TODO();
				return nullptr;
			},
		},
		value);
}

ast::Value *MLIRGenerator::visit(const ast::Delete *node)
{
	for (const auto &target : node->targets()) { target->codegen(this); }
	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::Dict *node)
{
	std::vector<MLIRValue *> keys;
	std::vector<MLIRValue *> values;
	std::vector<bool> requires_expansion;

	auto none = new_value(m_context.builder().create<mlir::py::ConstantOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()),
		m_context.builder().getNoneType()));

	for (const auto &[key, value] : llvm::zip(node->keys(), node->values())) {
		if (key) {
			keys.push_back(static_cast<MLIRValue *>(key->codegen(this)));
			requires_expansion.push_back(false);
		} else {
			keys.push_back(none);
			requires_expansion.push_back(true);
		}
		values.push_back(static_cast<MLIRValue *>(value->codegen(this)));
	}

	return build_dict(keys, values, requires_expansion, node->source_location());
}

ast::Value *MLIRGenerator::visit(const ast::DictComp *node)
{
	return build_comprehension(
		"<dictcomp>",
		[this, node]() { return build_dict({}, {}, node->source_location()); },
		[this, node](MLIRValue *container) {
			auto key = static_cast<MLIRValue *>(node->key()->codegen(this))->value;
			auto value = static_cast<MLIRValue *>(node->value()->codegen(this))->value;
			m_context.builder().create<mlir::py::DictAddOp>(
				loc(m_context.builder(), m_context.filename(), node->key()->source_location()),
				container->value,
				key,
				value);
		},
		node->generators(),
		node->source_location());
}

ast::Value *MLIRGenerator::visit(const ast::ExceptHandler *node)
{
	TODO();
	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::Expression *node)
{
	TODO();
	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::For *node)
{
	auto *parent = m_context.builder().getBlock()->getParent();

	auto iterable = static_cast<MLIRValue *>(node->iter()->codegen(this))->value;
	auto for_loop = m_context.builder().create<mlir::py::ForLoopOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()), iterable);
	auto &body_start = for_loop.getBody().emplaceBlock();

	auto &orelse = for_loop.getOrelse();
	auto *orelse_block =
		node->orelse().empty() ? nullptr : m_context.builder().createBlock(&orelse);

	m_context.builder().setInsertionPointToStart(&for_loop.getStep().emplaceBlock());
	auto iterator = new_value(for_loop.getStep().addArgument(
		m_context->pyobject_type(), m_context.builder().getUnknownLoc()));

	assign(node->target(), iterator, node->target()->source_location());
	m_context.builder().create<mlir::py::ControlFlowYield>(m_context.builder().getUnknownLoc());

	m_context.builder().setInsertionPointToStart(&body_start);
	for (const auto &el : node->body()) { el->codegen(this); }
	if (m_context.builder().getInsertionBlock()->empty()
		|| !m_context.builder()
				.getInsertionBlock()
				->back()
				.hasTrait<mlir::OpTrait::IsTerminator>()) {
		m_context.builder().create<mlir::py::ControlFlowYield>(m_context.builder().getUnknownLoc());
	}

	if (!node->orelse().empty()) {
		m_context.builder().setInsertionPointToStart(orelse_block);
		for (const auto &el : node->orelse()) { el->codegen(this); }
		if (m_context.builder().getBlock()->empty()
			|| !m_context.builder().getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
			m_context.builder().create<mlir::py::ControlFlowYield>(loc(
				m_context.builder(), m_context.filename(), node->body().back()->source_location()));
		}
	}

	m_context.builder().setInsertionPointAfter(for_loop);

	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::FormattedValue *node)
{
	if (node->format_spec()) { TODO(); }
	auto *value = static_cast<MLIRValue *>(node->value()->codegen(this));
	ASSERT(value);
	return new_value(m_context.builder().create<mlir::py::FormatValueOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()),
		m_context->pyobject_type(),
		value->value,
		mlir::py::FormatValueConversionAttr::get(&m_context.ctx(),
			mlir::py::FormatValueConversion{ static_cast<uint8_t>(node->conversion()) })));
}

ast::Value *MLIRGenerator::visit(const ast::FunctionDefinition *node)
{
	const std::string &mangled_name = Mangler::default_mangler().function_mangle(
		mangle_namespace(m_scope), node->name(), node->source_location());
	make_function(node->name(),
		mangled_name,
		node->args(),
		node->body(),
		node->decorator_list(),
		false,
		false,
		node->source_location());
	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::GeneratorExp *node)
{
	return build_comprehension(
		"<genexpr>",
		[this, node]() { return nullptr; },
		[this, node](MLIRValue *) {
			auto result = static_cast<MLIRValue *>(node->elt()->codegen(this))->value;
			m_context.builder().create<mlir::py::YieldOp>(
				loc(m_context.builder(), m_context.filename(), node->elt()->source_location()),
				m_context->pyobject_type(),
				result);
		},
		node->generators(),
		node->source_location());
}

ast::Value *MLIRGenerator::visit(const ast::Global *)
{
	// noop: this is handled by symbol scope resolution
	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::If *node)
{
	auto test = static_cast<MLIRValue *>(node->test()->codegen(this))->value;
	auto cond = m_context.builder().create<mlir::py::CastToBoolOp>(
		loc(m_context.builder(), m_context.filename(), node->test()->source_location()),
		m_context.builder().getI1Type(),
		test);
	auto *parent = cond.getOperation()->getParentRegion();

	auto if_block = m_context.builder().createBlock(parent);
	auto orelse_block = m_context.builder().createBlock(parent);
	auto continuation = m_context.builder().createBlock(parent);

	m_context.builder().setInsertionPointToEnd(cond.getOperation()->getBlock());
	m_context.builder().create<mlir::cf::CondBranchOp>(
		loc(m_context.builder(), m_context.filename(), node->test()->source_location()),
		cond,
		if_block,
		orelse_block);

	m_context.builder().setInsertionPointToStart(if_block);
	for (const auto &el : node->body()) { el->codegen(this); }
	if (m_context.builder().getBlock()->empty()
		|| !m_context.builder().getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
		m_context.builder().create<mlir::cf::BranchOp>(
			loc(m_context.builder(), m_context.filename(), node->body().back()->source_location()),
			continuation);
	}
	if (!node->orelse().empty()) {
		m_context.builder().setInsertionPointToStart(orelse_block);
		for (const auto &el : node->orelse()) { el->codegen(this); }
		if (m_context.builder().getBlock()->empty()
			|| !m_context.builder().getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
			m_context.builder().create<mlir::cf::BranchOp>(
				loc(m_context.builder(),
					m_context.filename(),
					node->orelse().back()->source_location()),
				continuation);
		}
	} else {
		orelse_block->replaceAllUsesWith(continuation);
		orelse_block->erase();
	}

	m_context.builder().setInsertionPointToStart(continuation);

	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::IfExpr *node)
{
	auto test = static_cast<MLIRValue *>(node->test()->codegen(this))->value;
	auto cond = m_context.builder().create<mlir::py::CastToBoolOp>(
		loc(m_context.builder(), m_context.filename(), node->test()->source_location()),
		m_context.builder().getI1Type(),
		test);
	auto *parent = cond.getOperation()->getParentRegion();

	auto if_block = m_context.builder().createBlock(parent);
	auto orelse_block = m_context.builder().createBlock(parent);
	auto continuation = m_context.builder().createBlock(parent);
	continuation->addArgument(m_context->pyobject_type(),
		loc(m_context.builder(), m_context.filename(), node->source_location()));

	m_context.builder().setInsertionPointToEnd(cond.getOperation()->getBlock());
	m_context.builder().create<mlir::cf::CondBranchOp>(
		loc(m_context.builder(), m_context.filename(), node->test()->source_location()),
		cond,
		if_block,
		orelse_block);

	m_context.builder().setInsertionPointToStart(if_block);
	auto true_case = static_cast<MLIRValue *>(node->body()->codegen(this))->value;
	m_context.builder().create<mlir::cf::BranchOp>(
		loc(m_context.builder(), m_context.filename(), node->body()->source_location()),
		continuation,
		mlir::ValueRange{ true_case });

	m_context.builder().setInsertionPointToStart(orelse_block);
	auto false_case = static_cast<MLIRValue *>(node->orelse()->codegen(this))->value;
	m_context.builder().create<mlir::cf::BranchOp>(
		loc(m_context.builder(), m_context.filename(), node->orelse()->source_location()),
		continuation,
		mlir::ValueRange{ false_case });

	m_context.builder().setInsertionPointToStart(continuation);

	return new_value(continuation->getArgument(0));
}

ast::Value *MLIRGenerator::visit(const ast::Import *node)
{
	for (const auto &n : node->names()) {
		// empty from_list
		auto from_list = mlir::DenseStringElementsAttr::get(
			mlir::VectorType::get({ 0 }, mlir::StringAttr::get(&m_context.ctx()).getType()), {});
		const uint32_t level = 0;

		auto module = new_value(m_context.builder().create<mlir::py::ImportOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			n.name,
			from_list,
			level));

		if (!n.asname.empty()) {
			store_name(n.asname, module, node->source_location());
		} else {
			if (const auto idx = n.name.find('.'); idx != std::string::npos) {
				const auto varname = n.name.substr(0, idx);
				store_name(varname, module, node->source_location());
			} else {
				store_name(n.name, module, node->source_location());
			}
		}
	}
	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::ImportFrom *node)
{
	std::vector<mlir::StringRef> names;
	names.reserve(node->names().size());
	for (const auto &n : node->names()) { names.emplace_back(n.name); }

	auto from_list = mlir::DenseStringElementsAttr::get(
		mlir::VectorType::get({ static_cast<int64_t>(names.size()) },
			mlir::StringAttr::get(&m_context.ctx()).getType()),
		names);

	auto module = m_context.builder().create<mlir::py::ImportOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()),
		m_context->pyobject_type(),
		node->module(),
		from_list,
		node->level());

	for (const auto &n : node->names()) {
		if (n.name == "*") {
			m_context.builder().create<mlir::py::ImportAllOp>(
				loc(m_context.builder(), m_context.filename(), node->source_location()), module);
		} else {
			auto imported_object = m_context.builder().create<mlir::py::ImportFromOp>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				module,
				n.name);
			if (n.asname.empty()) {
				store_name(n.name, new_value(imported_object), node->source_location());
			} else {
				store_name(n.asname, new_value(imported_object), node->source_location());
			}
		}
	}

	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::JoinedStr *node)
{
	py::String current_string;
	std::vector<mlir::Value> strings;
	for (const auto &value : node->values()) {
		if (auto c = as<ast::Constant>(value);
			c && std::holds_alternative<py::String>(*c->value())) {
			current_string.s += std::get<py::String>(*as<ast::Constant>(value)->value()).s;
		} else {
			if (!current_string.s.empty()) {
				strings.push_back(load_const(m_context.builder(),
					current_string.s,
					m_context.filename(),
					value->source_location()));
				current_string.s.clear();
			}
			ASSERT(as<ast::FormattedValue>(value));
			auto *str_value = value->codegen(this);
			ASSERT(str_value);
			strings.push_back(static_cast<MLIRValue &>(*str_value).value);
		}
	}
	if (!current_string.s.empty()) {
		strings.push_back(load_const(
			m_context.builder(), current_string.s, m_context.filename(), node->source_location()));
	}
	return new_value(m_context.builder().create<mlir::py::BuildStringOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()),
		m_context->pyobject_type(),
		strings));
}

ast::Value *MLIRGenerator::visit(const ast::Keyword *node) { return node->value()->codegen(this); }

ast::Value *MLIRGenerator::visit(const ast::Lambda *node)
{
	const std::string &mangled_name = Mangler::default_mangler().function_mangle(
		mangle_namespace(m_scope), "<lambda>", node->source_location());
	auto *fn = make_function("<lambda>",
		mangled_name,
		node->args(),
		{ std::make_shared<ast::Return>(node->body(), node->body()->source_location()) },
		{},
		true,
		false,
		node->source_location());
	ASSERT(fn);
	return fn;
}

ast::Value *MLIRGenerator::visit(const ast::List *node)
{
	std::vector<MLIRValue *> values;
	values.reserve(node->elements().size());
	auto requires_expansion = [](const std::shared_ptr<ast::ASTNode> &node) {
		return node->node_type() == ast::ASTNodeType::Starred;
	};
	std::vector<bool> value_requires_expansion(node->elements().size(), false);
	for (const auto &p : llvm::enumerate(node->elements())) {
		values.push_back(static_cast<MLIRValue *>(p.value()->codegen(this)));
		value_requires_expansion[p.index()] = requires_expansion(p.value());
	}

	return build_list(
		std::move(values), std::move(value_requires_expansion), node->source_location());
}

ast::Value *MLIRGenerator::visit(const ast::ListComp *node)
{
	return build_comprehension(
		"<listcomp>",
		[this, node]() { return build_list({}, node->source_location()); },
		[this, node](MLIRValue *container) {
			auto result = static_cast<MLIRValue *>(node->elt()->codegen(this))->value;
			m_context.builder().create<mlir::py::ListAppendOp>(
				loc(m_context.builder(), m_context.filename(), node->elt()->source_location()),
				container->value,
				result);
		},
		node->generators(),
		node->source_location());
}

ast::Value *MLIRGenerator::visit(const ast::Module *m)
{
	m_context.module()->setLoc(loc(m_context.builder(), m->filename(), SourceLocation{ 0, 0 }));
	const auto filename = fs::path(m->filename()).stem();
	[[maybe_unused]] auto module_scope = create_nested_scope(filename, filename);
	m_context.builder().setInsertionPointToEnd(m_context.module().getBody());
	auto module_fn =
		m_context.builder().create<mlir::func::FuncOp>(m_context.builder().getUnknownLoc(),
			"__hidden_init__",
			m_context.builder().getFunctionType({}, { m_context->pyobject_type() }));
	module_fn.setPrivate();
	auto *entry_block = module_fn.addEntryBlock();
	auto *exit_block = module_fn.addBlock();
	m_context.builder().setInsertionPointToEnd(entry_block);
	for (const auto &node : m->body()) { node->codegen(this); }

	// If a program does not end with a terminator instruction, jump to the exit_block
	if (m_context.builder().getBlock()->empty()
		|| !m_context.builder().getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
		m_context.builder().create<mlir::cf::BranchOp>(
			m_context.builder().getUnknownLoc(), exit_block);
	}

	m_context.builder().setInsertionPointToEnd(exit_block);
	auto result = m_context.builder().create<mlir::py::ConstantOp>(
		m_context.builder().getUnknownLoc(), m_context.builder().getNoneType());
	m_context.builder().create<mlir::func::ReturnOp>(
		m_context.builder().getUnknownLoc(), mlir::ValueRange{ result });
	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::NamedExpr *node)
{
	ASSERT(as<ast::Name>(node->target()));
	ASSERT(as<ast::Name>(node->target())->context_type() == ast::ContextType::STORE);
	ASSERT(as<ast::Name>(node->target())->ids().size() == 1);

	auto *src = node->value()->codegen(this);
	ASSERT(src);

	store_name(as<ast::Name>(node->target())->ids().front(),
		static_cast<MLIRValue *>(src),
		node->source_location());

	return src;
}

ast::Value *MLIRGenerator::visit(const ast::Name *node)
{
	ASSERT(node->ids().size() == 1);
	const auto name = node->ids()[0];
	if (node->context_type() == ast::ContextType::LOAD) {
		// return new_value(m_context.builder().create<mlir::py::LoadNameOp>(
		// 	loc(m_context.builder(), m_context.filename(), node->source_location()),
		// 	m_context->pyobject_type(),// TODO: propagate type information
		// 	name));
		return load_name(name, node->source_location());
	} else if (node->context_type() == ast::ContextType::DELETE) {
		delete_name(name, node->source_location());
		return nullptr;
	} else {
		TODO();
	}
}

ast::Value *MLIRGenerator::visit(const ast::NonLocal *) { return nullptr; }

ast::Value *MLIRGenerator::visit(const ast::Pass *) { return nullptr; }

ast::Value *MLIRGenerator::visit(const ast::Raise *node)
{
	if (node->cause()) {
		auto exception = static_cast<const MLIRValue &>(*node->exception()->codegen(this)).value;
		auto cause = static_cast<const MLIRValue &>(*node->cause()->codegen(this)).value;
		m_context.builder().create<mlir::py::RaiseOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			exception,
			cause);
	} else if (node->exception()) {
		auto exception = static_cast<const MLIRValue &>(*node->exception()->codegen(this)).value;
		m_context.builder().create<mlir::py::RaiseOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()), exception);
	} else {
		m_context.builder().create<mlir::py::RaiseOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()));
	}

	m_context.builder().createBlock(m_context.builder().getBlock()->getParent());
	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::Return *node)
{
	auto value = [this, node]() -> mlir::Value {
		if (node->value()) {
			return static_cast<const MLIRValue &>(*node->value()->codegen(this)).value;
		}
		return m_context.builder().create<mlir::py::ConstantOp>(
			m_context.builder().getUnknownLoc(), m_context.builder().getNoneType());
	}();

	return_value(new_value(value), node->source_location());

	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::Set *node)
{
	std::vector<MLIRValue *> elements;
	elements.reserve(node->elements().size());
	std::vector<bool> requires_expansion;
	requires_expansion.reserve(node->elements().size());
	for (const auto &el : node->elements()) {
		requires_expansion.emplace_back(as<ast::Starred>(el));
		elements.push_back(static_cast<MLIRValue *>(el->codegen(this)));
	}
	return build_set(std::move(elements), std::move(requires_expansion), node->source_location());
}

codegen::MLIRGenerator::MLIRValue *MLIRGenerator::build_comprehension(
	std::string_view function_name,
	std::function<MLIRValue *()> container_factory,
	std::function<void(MLIRValue *)> container_update,
	const std::vector<std::shared_ptr<ast::Comprehension>> &generators,
	const SourceLocation &source_location)
{
	const std::string &mangled_name = Mangler::default_mangler().function_mangle(
		mangle_namespace(m_scope), std::string{ function_name }, source_location);

	auto *last_block = m_context.builder().getBlock();

	m_context.builder().setInsertionPointToEnd(m_context.module().getBody(0));
	auto func_type = mlir::FunctionType::get(
		&m_context.ctx(), { m_context->pyobject_type() }, { m_context->pyobject_type() });
	std::vector<mlir::DictionaryAttr> args_attrs;
	std::vector<mlir::NamedAttribute> arg_attrs;
	arg_attrs.push_back(
		m_context.builder().getNamedAttr("llvm.name", m_context.builder().getStringAttr(".0")));
	args_attrs.push_back(m_context.builder().getDictionaryAttr(arg_attrs));
	auto f = m_context.builder().create<mlir::func::FuncOp>(
		loc(m_context.builder(), m_context.filename(), source_location),
		mangled_name,
		func_type,
		mlir::ArrayRef<mlir::NamedAttribute>{},
		args_attrs);
	f.setPrivate();

	std::vector<std::string> captures;
	{
		auto *entry_block = f.addEntryBlock();
		[[maybe_unused]] auto function_scope =
			setup_function(f, std::string{ function_name }, mangled_name);

		for (const auto &el : m_variable_visibility.at(mangled_name)->symbol_map.symbols) {
			if (el.visibility == VariablesResolver::Visibility::CELL
				&& m_variable_visibility.at(mangled_name)->captures.contains(el.name)) {
				captures.push_back(el.name);
			}
		}

		for (const auto &el : m_variable_visibility.at(mangled_name)->symbol_map.symbols) {
			if (el.visibility == VariablesResolver::Visibility::FREE
				&& m_variable_visibility.at(mangled_name)->captures.contains(el.name)) {
				captures.push_back(el.name);
			}
		}

		auto next_generator = [this](mlir::Value iterable,
								  const std::shared_ptr<ast::Comprehension> &generator) {
			auto for_loop = m_context.builder().create<mlir::py::ForLoopOp>(
				loc(m_context.builder(), m_context.filename(), generator->source_location()),
				iterable);
			m_context.builder().create<mlir::py::ControlFlowYield>(
				loc(m_context.builder(), m_context.filename(), generator->source_location()));
			// iterator
			{
				m_context.builder().setInsertionPointToStart(&for_loop.getStep().emplaceBlock());
				auto iterator = new_value(for_loop.getStep().addArgument(
					m_context->pyobject_type(), m_context.builder().getUnknownLoc()));
				assign(generator->target(), iterator, generator->target()->source_location());
				m_context.builder().create<mlir::py::ControlFlowYield>(
					m_context.builder().getUnknownLoc());
			}
			// loop body
			{
				auto &body_start = for_loop.getBody().emplaceBlock();

				m_context.builder().setInsertionPointToStart(&body_start);

				if (!generator->ifs().empty()) {
					mlir::Block *next{ nullptr };
					auto &body_continue = for_loop.getBody().emplaceBlock();
					auto &body_end = for_loop.getBody().emplaceBlock();

					for (const auto &el :
						generator->ifs() | std::views::take(generator->ifs().size() - 1)) {
						auto *current = m_context.builder().getInsertionBlock();

						next = m_context.builder().createBlock(&body_continue);

						m_context.builder().setInsertionPointToStart(current);
						auto cond = m_context.builder().create<mlir::py::CastToBoolOp>(
							loc(m_context.builder(), m_context.filename(), el->source_location()),
							m_context.builder().getI1Type(),
							static_cast<MLIRValue *>(el->codegen(this))->value);

						m_context.builder().create<mlir::cf::CondBranchOp>(
							loc(m_context.builder(), m_context.filename(), el->source_location()),
							cond,
							next,
							&body_continue);

						m_context.builder().setInsertionPointToStart(next);
					}
					{
						const auto &el = generator->ifs().back();
						auto cond = m_context.builder().create<mlir::py::CastToBoolOp>(
							loc(m_context.builder(), m_context.filename(), el->source_location()),
							m_context.builder().getI1Type(),
							static_cast<MLIRValue *>(el->codegen(this))->value);
						m_context.builder().create<mlir::cf::CondBranchOp>(
							loc(m_context.builder(), m_context.filename(), el->source_location()),
							cond,
							&body_end,
							&body_continue);
					}
					m_context.builder().setInsertionPointToStart(&body_continue);
					m_context.builder().create<mlir::py::ControlFlowYield>(
						loc(m_context.builder(),
							m_context.filename(),
							generator->source_location()),
						mlir::py::LoopOpKindAttr::get(
							&m_context.ctx(), mlir::py::LoopOpKind::continue_));
					m_context.builder().setInsertionPointToStart(&body_end);
				}
			}
		};

		const auto &first_generator = generators.front();

		m_context.builder().setInsertionPointToStart(entry_block);
		auto *container = container_factory();
		auto iterable = m_context.builder().create<mlir::py::LoadFastOp>(
			loc(m_context.builder(),
				m_context.filename(),
				first_generator->iter()->source_location()),
			m_context->pyobject_type(),
			".0");

		next_generator(iterable, first_generator);
		for (const auto &generator : generators | std::ranges::views::drop(1)) {
			auto iter = static_cast<MLIRValue &>(*generator->iter()->codegen(this)).value;
			next_generator(iter, generator);
		}
		container_update(container);
		m_context.builder().create<mlir::py::ControlFlowYield>(m_context.builder().getUnknownLoc());

		m_context.builder().setInsertionPointToEnd(entry_block);
		m_context.builder().getBlock()->back().erase();
		if (container) {
			return_value(container, first_generator->target()->source_location());
		} else {
			f->setAttr("is_generator", m_context.builder().getBoolAttr(true));
			auto none = m_context.builder().create<mlir::py::ConstantOp>(
				loc(m_context.builder(),
					m_context.filename(),
					first_generator->target()->source_location()),
				m_context.builder().getNoneType());
			return_value(new_value(none), first_generator->target()->source_location());
		}
	}

	m_context.builder().setInsertionPointToEnd(last_block);

	std::vector<mlir::StringRef> captures_ref;
	captures_ref.reserve(captures.size());
	for (const auto &el : captures) { captures_ref.push_back(el); }
	auto fn_obj = m_context.builder().create<mlir::py::MakeFunctionOp>(
		loc(m_context.builder(), m_context.filename(), source_location),
		m_context->pyobject_type(),
		mangled_name,
		mlir::ValueRange{},
		mlir::ValueRange{},
		mlir::DenseStringElementsAttr::get(
			mlir::VectorType::get({ static_cast<int64_t>(captures.size()) },
				mlir::StringAttr::get(&m_context.ctx()).getType()),
			captures_ref));

	auto iterable = static_cast<MLIRValue *>(generators.front()->iter()->codegen(this))->value;

	return new_value(m_context.builder().create<mlir::py::FunctionCallOp>(
		loc(m_context.builder(), m_context.filename(), source_location),
		m_context->pyobject_type(),
		fn_obj,
		mlir::ValueRange{ iterable },
		mlir::DenseStringElementsAttr::get(
			mlir::VectorType::get({ 0 }, mlir::StringAttr::get(&m_context.ctx()).getType()), {}),
		mlir::ValueRange{},
		false,
		false));
}

ast::Value *MLIRGenerator::visit(const ast::SetComp *node)
{
	return build_comprehension(
		"<setcomp>",
		[this, node]() { return build_set({}, node->source_location()); },
		[this, node](MLIRValue *container) {
			auto result = static_cast<MLIRValue *>(node->elt()->codegen(this))->value;
			m_context.builder().create<mlir::py::SetAddOp>(
				loc(m_context.builder(), m_context.filename(), node->elt()->source_location()),
				container->value,
				result);
		},
		node->generators(),
		node->source_location());
}

ast::Value *MLIRGenerator::visit(const ast::Starred *node)
{
	if (node->ctx() != ast::ContextType::LOAD) { TODO(); }
	return node->value()->codegen(this);
}

codegen::MLIRGenerator::MLIRValue *MLIRGenerator::build_slice(
	const ast::Subscript::SliceType &sliceNode,
	const SourceLocation &location)
{
	return std::visit(
		overloaded{
			[this](ast::Subscript::Index idx) -> MLIRValue * {
				return new_value(static_cast<const MLIRValue &>(*idx.value->codegen(this)).value);
			},
			[this, location](ast::Subscript::Slice slice) -> MLIRValue * {
				auto lower = slice.lower
								 ? static_cast<MLIRValue &>(*slice.lower->codegen(this)).value
								 : m_context.builder().create<mlir::py::ConstantOp>(
									   loc(m_context.builder(), m_context.filename(), location),
									   m_context.builder().getNoneType());
				auto upper = slice.upper
								 ? static_cast<MLIRValue &>(*slice.upper->codegen(this)).value
								 : m_context.builder().create<mlir::py::ConstantOp>(
									   loc(m_context.builder(), m_context.filename(), location),
									   m_context.builder().getNoneType());
				auto step = slice.step
								? static_cast<MLIRValue &>(*slice.step->codegen(this)).value
								: m_context.builder().create<mlir::py::ConstantOp>(
									  loc(m_context.builder(), m_context.filename(), location),
									  m_context.builder().getNoneType());
				return new_value(m_context.builder().create<mlir::py::BuildSliceOp>(
					loc(m_context.builder(), m_context.filename(), location),
					m_context->pyobject_type(),
					lower,
					upper,
					step));
			},
			[this, &location](ast::Subscript::ExtSlice slices) -> MLIRValue * {
				std::vector<MLIRValue *> slice_objects;
				slice_objects.reserve(slices.dims.size());
				for (const auto &slice : slices.dims) {
					auto *slice_value = std::visit(overloaded{ [this, &location](auto s) {
						return build_slice(s, location);
					} },
						slice);
					slice_objects.push_back(slice_value);
				}
				return build_tuple(slice_objects, location);
			},
		},
		sliceNode);
}

codegen::MLIRGenerator::MLIRValue *MLIRGenerator::build_list(
	const std::vector<codegen::MLIRGenerator::MLIRValue *> &els,
	const SourceLocation &location)
{
	std::vector<bool> requires_expansion(els.size(), false);
	return build_list(els, std::move(requires_expansion), location);
}

codegen::MLIRGenerator::MLIRValue *MLIRGenerator::build_list(
	const std::vector<codegen::MLIRGenerator::MLIRValue *> &els,
	std::vector<bool> requires_expansion,
	const SourceLocation &location)
{
	std::vector<mlir::Value> elements;
	elements.reserve(els.size());
	std::transform(els.begin(), els.end(), std::back_inserter(elements), [](MLIRValue *el) {
		return el->value;
	});

	llvm::SmallVector<bool> requires_expansion_;
	requires_expansion_.reserve(requires_expansion.size());
	std::transform(requires_expansion.begin(),
		requires_expansion.end(),
		std::back_inserter(requires_expansion_),
		[](bool el) -> int8_t { return el; });

	return new_value(m_context.builder().create<mlir::py::BuildListOp>(
		loc(m_context.builder(), m_context.filename(), location),
		m_context->pyobject_type(),
		elements,
		mlir::DenseBoolArrayAttr::get(&m_context.ctx(), std::move(requires_expansion_))));
}

codegen::MLIRGenerator::MLIRValue *MLIRGenerator::build_dict(
	const std::vector<codegen::MLIRGenerator::MLIRValue *> &keys,
	const std::vector<codegen::MLIRGenerator::MLIRValue *> &values,
	const SourceLocation &location)
{
	ASSERT(keys.size() == values.size());
	std::vector<bool> requires_expansion(keys.size(), false);
	return build_dict(keys, values, std::move(requires_expansion), location);
}

codegen::MLIRGenerator::MLIRValue *MLIRGenerator::build_dict(
	const std::vector<codegen::MLIRGenerator::MLIRValue *> &keys,
	const std::vector<codegen::MLIRGenerator::MLIRValue *> &values,
	std::vector<bool> requires_expansion,
	const SourceLocation &location)
{
	std::vector<mlir::Value> ks;
	ks.reserve(keys.size());
	std::transform(
		keys.begin(), keys.end(), std::back_inserter(ks), [](MLIRValue *el) { return el->value; });

	std::vector<mlir::Value> vs;
	vs.reserve(values.size());
	std::transform(values.begin(), values.end(), std::back_inserter(vs), [](MLIRValue *el) {
		return el->value;
	});

	llvm::SmallVector<bool> requires_expansion_;
	requires_expansion_.reserve(requires_expansion.size());
	std::transform(requires_expansion.begin(),
		requires_expansion.end(),
		std::back_inserter(requires_expansion_),
		[](bool el) -> int8_t { return el; });

	return new_value(m_context.builder().create<mlir::py::BuildDictOp>(
		loc(m_context.builder(), m_context.filename(), location),
		m_context->pyobject_type(),
		ks,
		vs,
		mlir::DenseBoolArrayAttr::get(&m_context.ctx(), std::move(requires_expansion_))));
}

codegen::MLIRGenerator::MLIRValue *MLIRGenerator::build_tuple(
	const std::vector<codegen::MLIRGenerator::MLIRValue *> &els,
	const SourceLocation &location)
{
	std::vector<bool> requires_expansion(els.size(), false);
	return build_tuple(els, std::move(requires_expansion), location);
}

codegen::MLIRGenerator::MLIRValue *MLIRGenerator::build_tuple(
	const std::vector<codegen::MLIRGenerator::MLIRValue *> &els,
	std::vector<bool> requires_expansion,
	const SourceLocation &location)
{
	std::vector<mlir::Value> elements;
	elements.reserve(els.size());
	std::transform(els.begin(), els.end(), std::back_inserter(elements), [](MLIRValue *el) {
		return el->value;
	});

	llvm::SmallVector<bool> requires_expansion_;
	requires_expansion_.reserve(requires_expansion.size());
	std::transform(requires_expansion.begin(),
		requires_expansion.end(),
		std::back_inserter(requires_expansion_),
		[](bool el) -> int8_t { return el; });

	return new_value(m_context.builder().create<mlir::py::BuildTupleOp>(
		loc(m_context.builder(), m_context.filename(), location),
		m_context->pyobject_type(),
		elements,
		mlir::DenseBoolArrayAttr::get(&m_context.ctx(), std::move(requires_expansion_))));
}

codegen::MLIRGenerator::MLIRValue *MLIRGenerator::build_set(
	std::vector<codegen::MLIRGenerator::MLIRValue *> els,
	const SourceLocation &location)
{
	std::vector<bool> requires_expansion(els.size(), false);
	return build_set(els, std::move(requires_expansion), location);
}

codegen::MLIRGenerator::MLIRValue *MLIRGenerator::build_set(std::vector<MLIRValue *> els,
	std::vector<bool> requires_expansion,
	const SourceLocation &location)
{
	std::vector<mlir::Value> elements;
	elements.reserve(els.size());
	std::transform(els.begin(), els.end(), std::back_inserter(elements), [](MLIRValue *el) {
		return el->value;
	});

	llvm::SmallVector<bool> requires_expansion_;
	requires_expansion_.reserve(requires_expansion.size());
	std::transform(requires_expansion.begin(),
		requires_expansion.end(),
		std::back_inserter(requires_expansion_),
		[](bool el) -> int8_t { return el; });

	return new_value(m_context.builder().create<mlir::py::BuildSetOp>(
		loc(m_context.builder(), m_context.filename(), location),
		m_context->pyobject_type(),
		std::move(elements),
		mlir::DenseBoolArrayAttr::get(&m_context.ctx(), std::move(requires_expansion_))));
}

void MLIRGenerator::return_value(MLIRValue *value, const SourceLocation &source_location)
{
	for (auto clear_exception : scope().clear_exception_before_return) {
		if (clear_exception) {
			m_context.builder().create<mlir::py::ClearExceptionStateOp>(
				loc(m_context.builder(), m_context.filename(), source_location));
		}
	}

	if (!scope().finally_blocks.empty()) {
		// m_context.builder().create<mlir::cf::BranchOp>(
		// 	loc(m_context.builder(), m_context.filename(), node->source_location()),
		// scope().finally_blocks.top());
		const auto finally_blocks = scope().finally_blocks;
		scope().finally_blocks.pop_back();
		finally_blocks.back()(true);
		std::for_each(finally_blocks.rbegin() + 1, finally_blocks.rend(), [this](const auto &f) {
			scope().finally_blocks.pop_back();
			f(false);
		});
		scope().finally_blocks = std::move(finally_blocks);
	}

	m_context.builder().create<mlir::func::ReturnOp>(
		loc(m_context.builder(), m_context.filename(), source_location), value->value);
}

MLIRGenerator::RAIIScope MLIRGenerator::setup_function(mlir::func::FuncOp &f,
	const std::string &function_name,
	const std::string &mangled_name)
{
	auto function_scope = create_nested_scope(function_name, mangled_name);

	const auto &name_visibility_it = m_variable_visibility.find(mangled_name);
	ASSERT(name_visibility_it != m_variable_visibility.end());
	const auto &symbol_map = name_visibility_it->second->symbol_map;
	const bool is_generator = name_visibility_it->second->is_generator;

	for (const auto &symbol : symbol_map.symbols) {
		const auto &varname = symbol.name;
		const auto &v = symbol.visibility;
		if (v == VariablesResolver::Visibility::FREE) {
			add_free_variable(m_context.builder(), varname, f);
		} else if (v == VariablesResolver::Visibility::CELL) {
			add_cell_variable(m_context.builder(), varname, f);
		} else if (v == VariablesResolver::Visibility::LOCAL) {
			// TODO
		} else {
			// TODO: add to co_names
			// A tuple containing names used by the bytecode:
			//  * global variables,
			//  * functions
			//  * classes
			//	* attributes loaded from objects
		}
	}

	if (name_visibility_it->second->is_generator) {
		// TODO: should add a generator op
		f->setAttr("is_generator", m_context.builder().getBoolAttr(true));
	}

	return function_scope;
}

MLIRGenerator::MLIRValue *MLIRGenerator::make_function(const std::string &function_name,
	const std::string &mangled_name,
	const std::shared_ptr<ast::Arguments> &args,
	const std::vector<std::shared_ptr<ast::ASTNode>> &body,
	const std::vector<std::shared_ptr<ast::ASTNode>> &decorator_list,
	bool is_anon,
	bool is_async,
	const SourceLocation &source_location)
{
	auto *last_block = m_context.builder().getBlock();

	std::vector<mlir::Value> decorator_functions;
	decorator_functions.reserve(decorator_list.size());
	for (const auto &decorator_function : decorator_list) {
		auto *f = decorator_function->codegen(this);
		ASSERT(f);
		decorator_functions.push_back(static_cast<MLIRValue *>(f)->value);
	}

	const size_t args_size = args->args().size() + args->posonlyargs().size()
							 + args->kwonlyargs().size() + (args->vararg() != nullptr)
							 + (args->kwarg() != nullptr);

	std::vector<mlir::Value> defaults;
	for (const auto &default_ : args->defaults()) {
		defaults.push_back(static_cast<MLIRValue *>(default_->codegen(this))->value);
	}

	std::vector<mlir::Value> kw_defaults;
	kw_defaults.reserve(args->kw_defaults().size());
	for (const auto &default_ : args->kw_defaults()) {
		if (default_) {
			kw_defaults.push_back(static_cast<MLIRValue *>(default_->codegen(this))->value);
		}
	}

	std::vector<mlir::Type> param_types(args_size, m_context->pyobject_type());

	auto func_type =
		mlir::FunctionType::get(&m_context.ctx(), param_types, { m_context->pyobject_type() });
	std::vector<mlir::DictionaryAttr> args_attrs;
	for (const auto &arg : args->argument_names()) {
		std::vector<mlir::NamedAttribute> arg_attrs;
		arg_attrs.push_back(
			m_context.builder().getNamedAttr("llvm.name", m_context.builder().getStringAttr(arg)));
		args_attrs.push_back(m_context.builder().getDictionaryAttr(arg_attrs));
	}

	for (const auto &arg : args->kw_only_argument_names()) {
		std::vector<mlir::NamedAttribute> arg_attrs;
		arg_attrs.push_back(
			m_context.builder().getNamedAttr("llvm.name", m_context.builder().getStringAttr(arg)));
		arg_attrs.push_back(m_context.builder().getNamedAttr(
			"llvm.kwonlyarg", m_context.builder().getBoolAttr(true)));
		args_attrs.push_back(m_context.builder().getDictionaryAttr(arg_attrs));
	}

	if (args->vararg()) {
		std::vector<mlir::NamedAttribute> arg_attrs;
		arg_attrs.push_back(m_context.builder().getNamedAttr(
			"llvm.name", m_context.builder().getStringAttr(args->vararg()->name())));
		arg_attrs.push_back(
			m_context.builder().getNamedAttr("llvm.vararg", m_context.builder().getBoolAttr(true)));
		args_attrs.push_back(m_context.builder().getDictionaryAttr(arg_attrs));
	}

	if (args->kwarg()) {
		std::vector<mlir::NamedAttribute> arg_attrs;
		arg_attrs.push_back(m_context.builder().getNamedAttr(
			"llvm.name", m_context.builder().getStringAttr(args->kwarg()->name())));
		arg_attrs.push_back(
			m_context.builder().getNamedAttr("llvm.kwarg", m_context.builder().getBoolAttr(true)));
		args_attrs.push_back(m_context.builder().getDictionaryAttr(arg_attrs));
	}

	m_context.builder().setInsertionPointToEnd(
		&m_context.module().getBodyRegion().getBlocks().back());
	auto f = m_context.builder().create<mlir::func::FuncOp>(
		loc(m_context.builder(), m_context.filename(), source_location),
		mangled_name,
		func_type,
		mlir::ArrayRef<mlir::NamedAttribute>{},
		args_attrs);
	m_context.builder().setInsertionPointToStart(f.addEntryBlock());

	std::vector<std::string> captures;

	{
		[[maybe_unused]] auto function_scope = setup_function(f, function_name, mangled_name);
		if (is_async) { f->setAttr("async", m_context.builder().getBoolAttr(true)); }

		// captures.reserve(m_variable_visibility.at(mangled_name)->captures.size());
		for (const auto &el : m_variable_visibility.at(mangled_name)->symbol_map.symbols) {
			if (el.visibility == VariablesResolver::Visibility::CELL
				&& m_variable_visibility.at(mangled_name)->captures.contains(el.name)) {
				captures.push_back(el.name);
			}
		}

		for (const auto &el : m_variable_visibility.at(mangled_name)->symbol_map.symbols) {
			if (el.visibility == VariablesResolver::Visibility::FREE
				&& m_variable_visibility.at(mangled_name)->captures.contains(el.name)) {
				captures.push_back(el.name);
			}
		}

		m_context.builder().setInsertionPointToStart(&f.front());
		for (const auto &el : body) { el->codegen(this); }

		if (m_context.builder().getBlock()->empty()
			|| !m_context.builder().getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
			auto none = m_context.builder().create<mlir::py::ConstantOp>(
				m_context.builder().getUnknownLoc(), m_context.builder().getNoneType());
			return_value(new_value(none), source_location);
		}
	}

	m_context.builder().setInsertionPointToEnd(last_block);
	std::vector<mlir::StringRef> captures_ref;
	captures_ref.reserve(captures.size());
	for (const auto &el : captures) { captures_ref.push_back(el); }
	auto fn_obj = new_value(m_context.builder().create<mlir::py::MakeFunctionOp>(
		loc(m_context.builder(), m_context.filename(), source_location),
		m_context->pyobject_type(),
		mangled_name,
		defaults,
		kw_defaults,
		mlir::DenseStringElementsAttr::get(
			mlir::VectorType::get({ static_cast<int64_t>(captures.size()) },
				mlir::StringAttr::get(&m_context.ctx()).getType()),
			captures_ref)));

	if (is_anon) { return fn_obj; }
	store_name(function_name, fn_obj, source_location);

	if (!decorator_functions.empty()) {
		ASSERT(!is_anon);
		mlir::Value arg = load_name(function_name, source_location)->value;
		for (const auto &decorator_function : decorator_functions | std::ranges::views::reverse) {
			arg = m_context.builder().create<mlir::py::FunctionCallOp>(decorator_function.getLoc(),
				m_context->pyobject_type(),
				decorator_function,
				mlir::ValueRange{ arg },
				mlir::DenseStringElementsAttr::get(
					mlir::VectorType::get({ 0 }, mlir::StringAttr::get(&m_context.ctx()).getType()),
					{}),
				mlir::ValueRange{},
				false,
				false);
		}
		store_name(function_name, new_value(arg), source_location);
	}

	return nullptr;
}


ast::Value *MLIRGenerator::visit(const ast::Subscript *node)
{
	auto value = static_cast<const MLIRValue &>(*node->value()->codegen(this)).value;
	auto index = build_slice(node->slice(), node->source_location())->value;

	switch (node->context()) {
	case ast::ContextType::DELETE: {
		m_context.builder().create<mlir::py::DeleteSubscriptOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()), value, index);
		return nullptr;
	} break;
	case ast::ContextType::LOAD: {
		return new_value(m_context.builder().create<mlir::py::BinarySubscriptOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			value,
			index));
	} break;
	case ast::ContextType::STORE: {
		TODO();
	} break;
	case ast::ContextType::UNSET: {
		TODO();
	} break;
	}
	ASSERT_NOT_REACHED();
}

ast::Value *MLIRGenerator::visit(const ast::Try *node)
{
	auto *current = m_context.builder().getBlock();
	auto *parent = current->getParent();

	auto try_op = m_context.builder().create<mlir::py::TryOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()),
		node->handlers().size());

	if (!try_op.getHandlers().empty()) {
		scope().unhappy_path.push(&try_op.getHandlers().front().front());
	}

	if (!node->finalbody().empty()) { try_op.getFinally().emplaceBlock(); }
	if (!node->orelse().empty()) { try_op.getOrelse().emplaceBlock(); }

	scope().finally_blocks.emplace_back([this, node](bool first) {
		(void)first;
		// if (!first) { m_context.builder().create<mlir::py::LeaveExceptionHandling>(); }
		// auto current_fn = getParentOfType<mlir::func::FuncOp, mlir::py::ClassDefinitionOp>(
		// 	m_context.builder().getInsertionBlock()->getParent());
		auto *current = m_context.builder().getInsertionBlock();
		auto *final_block = m_context.builder().createBlock(current->getParent());
		m_context.builder().setInsertionPointToEnd(current);
		m_context.builder().create<mlir::cf::BranchOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()), final_block);
		m_context.builder().setInsertionPointToEnd(final_block);
		if (!node->finalbody().empty()) {
			for (auto el : node->finalbody()) { el->codegen(this); }
		}
		if (!m_context.builder().getBlock()->empty()
			&& m_context.builder().getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
			m_context.builder().createBlock(current->getParent());
		}
	});

	m_context.builder().setInsertionPointToStart(&try_op.getBody().emplaceBlock());
	for (const auto &el : node->body()) { el->codegen(this); }
	if (m_context.builder().getBlock()->empty()
		|| !m_context.builder().getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
		m_context.builder().create<mlir::py::ControlFlowYield>(
			loc(m_context.builder(), m_context.filename(), node->source_location()));
	}

	std::for_each(try_op.getHandlers().begin(),
		try_op.getHandlers().end(),
		[this](mlir::Region &handler_region) {
			ASSERT(handler_region.getBlocks().empty());
			handler_region.emplaceBlock();
		});

	for (auto p : llvm::enumerate(llvm::zip(try_op.getHandlers(), node->handlers()))) {
		const auto &idx = p.index();
		auto [handler_region, handler] = p.value();

		ASSERT(!handler_region.getBlocks().empty());

		m_context.builder().setInsertionPointToStart(&handler_region.front());
		auto handler_op = m_context.builder().create<mlir::py::TryHandlerScope>(
			loc(m_context.builder(), m_context.filename(), handler->source_location()));

		if (handler->type()) {
			auto *exception_check_block = m_context.builder().createBlock(&handler_op.getCond());
			auto exception_type = handler->type()->codegen(this);
			if (!handler->name().empty()) {
				store_name(handler->name(),
					static_cast<MLIRValue *>(exception_type),
					handler->source_location());
			}
			m_context.builder().create<mlir::py::ConditionOp>(
				loc(m_context.builder(), m_context.filename(), handler->source_location()),
				static_cast<MLIRValue *>(exception_type)->value);
		}

		m_context.builder().createBlock(&handler_op.getHandler());
		{
			ClearExceptionBeforeReturn clear_exception_before_return{ scope() };
			for (auto el : handler->body()) { el->codegen(this); }
			if (m_context.builder().getBlock()->empty()
				|| !m_context.builder()
						.getBlock()
						->back()
						.hasTrait<mlir::OpTrait::IsTerminator>()) {
				m_context.builder().create<mlir::py::ControlFlowYield>(
					loc(m_context.builder(), m_context.filename(), node->source_location()));
			}
		}
	}

	if (!node->orelse().empty()) {
		m_context.builder().setInsertionPointToStart(&try_op.getOrelse().front());
		for (auto el : node->orelse()) { el->codegen(this); }
		if (m_context.builder().getBlock()->empty()
			|| !m_context.builder().getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
			m_context.builder().create<mlir::py::ControlFlowYield>(
				loc(m_context.builder(), m_context.filename(), node->source_location()));
		}
	}

	scope().finally_blocks.pop_back();
	if (!node->handlers().empty()) { scope().unhappy_path.pop(); }

	if (!node->finalbody().empty()) {
		m_context.builder().setInsertionPointToStart(&try_op.getFinally().front());
		for (auto el : node->finalbody()) { el->codegen(this); }
		if (m_context.builder().getBlock()->empty()
			|| !m_context.builder().getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
			m_context.builder().create<mlir::py::ControlFlowYield>(
				loc(m_context.builder(), m_context.filename(), node->source_location()));
		}
	}
	m_context.builder().setInsertionPointAfter(try_op);

	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::Tuple *node)
{
	std::vector<MLIRValue *> values;
	values.reserve(node->elements().size());
	auto requires_expansion = [](const std::shared_ptr<ast::ASTNode> &node) {
		return node->node_type() == ast::ASTNodeType::Starred;
	};
	std::vector<bool> value_requires_expansion(node->elements().size(), false);
	bool delete_context = false;
	for (const auto &p : llvm::enumerate(node->elements())) {
		values.push_back(static_cast<MLIRValue *>(p.value()->codegen(this)));
		if (values.back() == nullptr) {
			delete_context = true;
			ASSERT(as<ast::Name>(p.value()));
			ASSERT(as<ast::Name>(p.value())->context_type() == ast::ContextType::DELETE);
		}
		value_requires_expansion[p.index()] = requires_expansion(p.value());
	}

	if (delete_context) {
		ASSERT(std::all_of(values.begin(), values.end(), [](auto *el) { return el == nullptr; }));
		return nullptr;
	}

	return build_tuple(
		std::move(values), std::move(value_requires_expansion), node->source_location());
}

ast::Value *MLIRGenerator::visit(const ast::UnaryExpr *node)
{
	auto src = static_cast<const MLIRValue &>(*node->operand()->codegen(this)).value;
	switch (node->op_type()) {
	case ast::UnaryOpType::ADD: {
		return new_value(m_context.builder().create<mlir::py::PositiveOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			src));
	} break;
	case ast::UnaryOpType::SUB: {
		return new_value(m_context.builder().create<mlir::py::NegativeOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			src));
	} break;
	case ast::UnaryOpType::INVERT: {
		return new_value(m_context.builder().create<mlir::py::InvertOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			src));
	} break;
	case ast::UnaryOpType::NOT: {
		return new_value(m_context.builder().create<mlir::py::NotOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()),
			m_context->pyobject_type(),
			src));
	} break;
	}
	ASSERT_NOT_REACHED();
}

ast::Value *MLIRGenerator::visit(const ast::While *node)
{
	auto *current_block = m_context.builder().getInsertionBlock();
	auto *parent = current_block->getParent();

	auto while_op = m_context.builder().create<mlir::py::WhileOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()));

	auto &condition = while_op.getCondition();
	auto &body = while_op.getBody();
	auto &orelse = while_op.getOrelse();

	auto *condition_block = m_context.builder().createBlock(&condition);
	auto *body_start_block = m_context.builder().createBlock(&body);
	auto *orelse_block =
		node->orelse().empty() ? nullptr : m_context.builder().createBlock(&orelse);

	m_context.builder().setInsertionPointToStart(condition_block);
	auto test = static_cast<MLIRValue *>(node->test()->codegen(this))->value;
	m_context.builder().create<mlir::py::ConditionOp>(test.getLoc(), test);

	m_context.builder().setInsertionPointToStart(body_start_block);
	for (const auto &el : node->body()) { el->codegen(this); }

	if (m_context.builder().getBlock()->empty()
		|| !m_context.builder().getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
		m_context.builder().create<mlir::py::ControlFlowYield>(
			loc(m_context.builder(), m_context.filename(), node->body().back()->source_location()));
	}

	if (!node->orelse().empty()) {
		m_context.builder().setInsertionPointToStart(orelse_block);
		for (const auto &el : node->orelse()) { el->codegen(this); }
		if (m_context.builder().getBlock()->empty()
			|| !m_context.builder().getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
			m_context.builder().create<mlir::py::ControlFlowYield>(loc(
				m_context.builder(), m_context.filename(), node->body().back()->source_location()));
		}
	}

	m_context.builder().setInsertionPointToEnd(current_block);

	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::With *node)
{
	if (node->items().size() > 1) { TODO(); }

	std::vector<mlir::Value> with_item_results;
	for (const auto &item : node->items()) {
		with_item_results.push_back(static_cast<MLIRValue &>(*item->codegen(this)).value);
	}

	auto *current_block = m_context.builder().getBlock();
	auto *parent = current_block->getParent();

	auto with = m_context.builder().create<mlir::py::WithOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()), with_item_results);

	auto with_exit_factory = [this, node, &with_item_results](bool first) {
		ASSERT(node->items().size() == 1);

		for (size_t i = 0; const auto &item : with_item_results) {
			if (!first) {
				// m_context.builder().create<mlir::py::LeaveExceptionHandling>(item.getLoc());
			}

			auto exit = m_context.builder().create<mlir::py::LoadMethodOp>(
				item.getLoc(), m_context->pyobject_type(), item, "__exit__");

			auto none = m_context.builder().create<mlir::py::ConstantOp>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context.builder().getNoneType());

			m_context.builder().create<mlir::py::FunctionCallOp>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context->pyobject_type(),
				exit,
				std::vector<mlir::Value>{ none, none, none },
				mlir::DenseStringElementsAttr::get(
					mlir::VectorType::get({ 0 }, mlir::StringAttr::get(&m_context.ctx()).getType()),
					{}),
				std::vector<mlir::Value>{},
				false,
				false);
		}

		m_context.builder().create<mlir::py::ClearExceptionStateOp>(
			loc(m_context.builder(), m_context.filename(), node->source_location()));
	};
	scope().finally_blocks.push_back(with_exit_factory);

	auto &body_start = with.getBody().emplaceBlock();
	m_context.builder().setInsertionPointToStart(&body_start);

	for (const auto &el : node->body()) { el->codegen(this); }
	if (m_context.builder().getBlock()->empty()
		|| !m_context.builder().getBlock()->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
		m_context.builder().create<mlir::py::ControlFlowYield>(
			loc(m_context.builder(), m_context.filename(), node->source_location()));
	}
	scope().finally_blocks.pop_back();

	m_context.builder().setInsertionPointToEnd(current_block);

	return nullptr;
}

ast::Value *MLIRGenerator::visit(const ast::WithItem *node)
{
	auto expr = static_cast<MLIRValue *>(node->context_expr()->codegen(this))->value;
	auto method = m_context.builder().create<mlir::py::LoadMethodOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()),
		m_context->pyobject_type(),
		expr,
		"__enter__");
	auto item_result = m_context.builder().create<mlir::py::FunctionCallOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()),
		m_context->pyobject_type(),
		method,
		std::vector<mlir::Value>{},
		mlir::DenseStringElementsAttr::get(
			mlir::VectorType::get({ 0 }, mlir::StringAttr::get(&m_context.ctx()).getType()), {}),
		std::vector<mlir::Value>{},
		false,
		false);

	if (auto optional_vars = node->optional_vars()) {
		assign(optional_vars, new_value(item_result), node->source_location());
	}

	return new_value(expr);
}

ast::Value *MLIRGenerator::visit(const ast::Yield *node)
{
	auto value = [this, node]() -> mlir::Value {
		if (node->value()) {
			return static_cast<MLIRValue *>(node->value()->codegen(this))->value;
		} else {
			return m_context.builder().create<mlir::py::ConstantOp>(
				loc(m_context.builder(), m_context.filename(), node->source_location()),
				m_context.builder().getNoneType());
		}
	}();
	return new_value(m_context.builder().create<mlir::py::YieldOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()),
		m_context->pyobject_type(),
		value));
}

ast::Value *MLIRGenerator::visit(const ast::YieldFrom *node)
{
	auto value = static_cast<MLIRValue *>(node->value()->codegen(this))->value;
	return new_value(m_context.builder().create<mlir::py::YieldFromOp>(
		loc(m_context.builder(), m_context.filename(), node->source_location()),
		m_context->pyobject_type(),
		value));
}

MLIRGenerator::RAIIScope MLIRGenerator::create_nested_scope(const std::string &name,
	const std::string &mangled_name)
{
	return RAIIScope{ m_scope.emplace_back(Scope{
						  .name = name,
						  .mangled_name = mangled_name,
						  .finally_blocks = {},
					  }),
		this };
}

std::string MLIRGenerator::mangle_namespace(const std::deque<MLIRGenerator::Scope> &s) const
{
	ASSERT(!s.empty());
	return std::accumulate(
		s.begin() + 1, s.end(), s.front().name, [](std::string acc, const Scope &s) {
			return acc + '.' + s.name;
		});
}

}// namespace codegen