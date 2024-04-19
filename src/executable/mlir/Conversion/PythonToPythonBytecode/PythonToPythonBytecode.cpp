#include "Conversion/PythonToPythonBytecode/PythonToPythonBytecode.hpp"
#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/IR/Dialect.hpp"
#include "Dialect/Python/IR/PythonAttributes.hpp"
#include "Dialect/Python/IR/PythonOps.hpp"
#include "ast/AST.hpp"
#include "executable/Mangler.hpp"
#include "executable/bytecode/instructions/BinaryOperation.hpp"
#include "executable/bytecode/instructions/GetAwaitable.hpp"
#include "executable/bytecode/instructions/Unary.hpp"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace py {

	namespace {
		struct ConstantLoadLowering : public mlir::OpRewritePattern<py::ConstantOp>
		{
			using OpRewritePattern<py::ConstantOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(py::ConstantOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto constant_value = op.getValue();


				auto ellipsis =
					mlir::detail::AttributeUniquer::get<mlir::py::EllipsisAttr>(getContext());
				if (op.getValue() == ellipsis) {
					rewriter.replaceOpWithNewOp<mlir::emitpybytecode::LoadEllipsisOp>(
						op, op.getOutput().getType());
					return success();
				}

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ConstantOp>(
					op, op.getOutput().getType(), constant_value);

				return success();
			}
		};

		struct StoreNameLowering : public mlir::OpRewritePattern<py::StoreNameOp>
		{
			using OpRewritePattern<py::StoreNameOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(py::StoreNameOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto object_name = op.getNameAttr();
				auto object_value = op.getValue();

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::StoreNameOp>(
					op, op.getOutput().getType(), object_name, object_value);

				return success();
			}
		};

		struct StoreDerefLowering : public mlir::OpRewritePattern<py::StoreDerefOp>
		{
			using OpRewritePattern<py::StoreDerefOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(py::StoreDerefOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto object_name = op.getNameAttr();
				auto object_value = op.getValue();

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::StoreDerefOp>(
					op, op.getOutput().getType(), object_name, object_value);

				return success();
			}
		};

		template<typename T> struct LocalDeclarationInterface : public T
		{
			template<typename... Args>
			LocalDeclarationInterface(Args &&...args) : T(std::forward<Args>(args)...)
			{}

			void addLocalIdentifierToParentFunction(mlir::func::FuncOp &fn,
				mlir::StringAttr identifier,
				mlir::OpBuilder &builder) const
			{
				if (fn->hasAttr("locals")) {
					auto names = fn->getAttr("locals");
					std::vector<StringRef> names_vec;
					auto arr = names.cast<mlir::ArrayAttr>().getValue();
					if (std::find_if(arr.begin(),
							arr.end(),
							[identifier](mlir::Attribute attr) {
								return attr.cast<mlir::StringAttr>().getValue() == identifier;
							})
						!= arr.end()) {
						return;
					}
					std::transform(arr.begin(),
						arr.end(),
						std::back_inserter(names_vec),
						[](mlir::Attribute attr) {
							return attr.cast<mlir::StringAttr>().getValue();
						});
					names_vec.emplace_back(identifier);
					fn->setAttr("locals", builder.getStrArrayAttr(names_vec));
				} else {
					fn->setAttr("locals", builder.getStrArrayAttr({ identifier }));
				}
			}
		};

		namespace {
			void add_identifier(mlir::func::FuncOp &fn,
				mlir::StringRef identifier,
				mlir::OpBuilder &builder)
			{
				if (fn->hasAttr("names")) {
					auto names = fn->getAttr("names");
					std::vector<StringRef> names_vec;
					auto arr = names.cast<mlir::ArrayAttr>().getValue();
					if (std::find_if(arr.begin(),
							arr.end(),
							[identifier](mlir::Attribute attr) {
								return attr.cast<mlir::StringAttr>().getValue() == identifier;
							})
						!= arr.end()) {
						return;
					}
					std::transform(arr.begin(),
						arr.end(),
						std::back_inserter(names_vec),
						[](mlir::Attribute attr) {
							return attr.cast<mlir::StringAttr>().getValue();
						});
					names_vec.emplace_back(identifier);
					fn->setAttr("names", builder.getStrArrayAttr(names_vec));
				} else {
					fn->setAttr("names", builder.getStrArrayAttr({ identifier }));
				}
			}
		}// namespace

		template<typename T> struct GlobalDeclarationInterface : public T
		{
			template<typename... Args>
			GlobalDeclarationInterface(Args &&...args) : T(std::forward<Args>(args)...)
			{}

			void addGlobalIdentifierToParentFunction(mlir::func::FuncOp &fn,
				mlir::StringAttr identifier,
				mlir::OpBuilder &builder) const
			{
				add_identifier(fn, identifier, builder);
			}
		};

		struct StoreFastLowering
			: public LocalDeclarationInterface<mlir::OpRewritePattern<py::StoreFastOp>>
		{
			using LocalDeclarationInterface<
				OpRewritePattern<py::StoreFastOp>>::LocalDeclarationInterface;

			mlir::LogicalResult matchAndRewrite(py::StoreFastOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto object_name = op.getNameAttr();
				auto object_value = op.getValue();

				auto parent = op.getOperation()->getParentOp();
				auto fn = mlir::cast_or_null<mlir::func::FuncOp>(parent);
				ASSERT(fn);
				addLocalIdentifierToParentFunction(fn, object_name, rewriter);

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::StoreFastOp>(
					op, op.getOutput().getType(), object_name, object_value);

				return success();
			}
		};

		struct StoreGlobalLowering
			: public GlobalDeclarationInterface<mlir::OpRewritePattern<py::StoreGlobalOp>>
		{
			using GlobalDeclarationInterface<
				OpRewritePattern<py::StoreGlobalOp>>::GlobalDeclarationInterface;

			mlir::LogicalResult matchAndRewrite(py::StoreGlobalOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto parent = op.getOperation()->getParentOp();
				auto fn = mlir::cast_or_null<mlir::func::FuncOp>(parent);
				ASSERT(fn);
				auto object_name = op.getNameAttr();
				addGlobalIdentifierToParentFunction(fn, object_name, rewriter);

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::StoreGlobalOp>(
					op, op.getOutput().getType(), object_name, op.getValue());

				return success();
			}
		};

		struct LoadNameLowering : public mlir::OpRewritePattern<py::LoadNameOp>
		{
			using OpRewritePattern<py::LoadNameOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(py::LoadNameOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto object_name = op.getNameAttr();

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::LoadNameOp>(
					op, op.getOutput().getType(), object_name);

				return success();
			}
		};


		struct LoadGlobalLowering
			: public GlobalDeclarationInterface<mlir::OpRewritePattern<py::LoadGlobalOp>>
		{
			using GlobalDeclarationInterface<
				OpRewritePattern<py::LoadGlobalOp>>::GlobalDeclarationInterface;

			mlir::LogicalResult matchAndRewrite(py::LoadGlobalOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto parent = op.getOperation()->getParentOp();
				auto fn = mlir::cast_or_null<mlir::func::FuncOp>(parent);
				ASSERT(fn);
				auto object_name = op.getNameAttr();
				addGlobalIdentifierToParentFunction(fn, object_name, rewriter);

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::LoadGlobalOp>(
					op, op.getOutput().getType(), object_name);

				return success();
			}
		};


		struct LoadFastLowering
			: public LocalDeclarationInterface<mlir::OpRewritePattern<py::LoadFastOp>>
		{
			using LocalDeclarationInterface<
				mlir::OpRewritePattern<py::LoadFastOp>>::LocalDeclarationInterface;

			mlir::LogicalResult matchAndRewrite(py::LoadFastOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto parent = op.getOperation()->getParentOp();
				auto fn = mlir::cast_or_null<mlir::func::FuncOp>(parent);
				ASSERT(fn);
				auto object_name = op.getNameAttr();
				addLocalIdentifierToParentFunction(fn, object_name, rewriter);

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::LoadFastOp>(
					op, op.getOutput().getType(), object_name);

				return success();
			}
		};

		struct LoadDerefLowering : public mlir::OpRewritePattern<py::LoadDerefOp>
		{
			using OpRewritePattern<py::LoadDerefOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(py::LoadDerefOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto parent = op.getOperation()->getParentOp();
				auto fn = mlir::cast_or_null<mlir::func::FuncOp>(parent);
				ASSERT(fn);
				auto object_name = op.getNameAttr();

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::LoadDerefOp>(
					op, op.getOutput().getType(), object_name);

				return success();
			}
		};

		struct DeleteNameLowering : public mlir::OpRewritePattern<py::DeleteNameOp>
		{
			using OpRewritePattern<py::DeleteNameOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(py::DeleteNameOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto object_name = op.getNameAttr();

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::DeleteNameOp>(op, object_name);

				return success();
			}
		};

		struct DeleteFastLowering
			: public LocalDeclarationInterface<mlir::OpRewritePattern<py::DeleteFastOp>>
		{
			using LocalDeclarationInterface<
				mlir::OpRewritePattern<py::DeleteFastOp>>::LocalDeclarationInterface;

			mlir::LogicalResult matchAndRewrite(py::DeleteFastOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto parent = op.getOperation()->getParentOp();
				auto fn = mlir::cast_or_null<mlir::func::FuncOp>(parent);
				ASSERT(fn);
				auto object_name = op.getNameAttr();
				addLocalIdentifierToParentFunction(fn, object_name, rewriter);

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::DeleteFastOp>(op, object_name);

				return success();
			}
		};

		struct DeleteGlobalLowering
			: public GlobalDeclarationInterface<mlir::OpRewritePattern<py::DeleteGlobalOp>>
		{
			using GlobalDeclarationInterface<
				OpRewritePattern<py::DeleteGlobalOp>>::GlobalDeclarationInterface;

			mlir::LogicalResult matchAndRewrite(py::DeleteGlobalOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto parent = op.getOperation()->getParentOp();
				auto fn = mlir::cast_or_null<mlir::func::FuncOp>(parent);
				ASSERT(fn);
				auto object_name = op.getNameAttr();
				addGlobalIdentifierToParentFunction(fn, object_name, rewriter);

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::DeleteGlobalOp>(op, object_name);

				return success();
			}
		};

		struct CallFunctionLowering : public mlir::OpRewritePattern<py::FunctionCallOp>
		{
			using OpRewritePattern<py::FunctionCallOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(py::FunctionCallOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto callee = op.getCallee();
				auto args = op.getArgs();

				if (op.getRequiresArgsExpansion() || op.getRequiresKwargsExpansion()) {
					ASSERT(args.size() <= 1);
					ASSERT(op.getKwargs().size() <= 1);
					rewriter.replaceOpWithNewOp<mlir::emitpybytecode::FunctionCallExOp>(op,
						op.getOutput().getType(),
						callee,
						op.getRequiresArgsExpansion() ? args.front() : nullptr,
						op.getRequiresKwargsExpansion() ? op.getKwargs().front() : nullptr);
				} else if (!op.getKeywords().empty()) {
					rewriter.replaceOpWithNewOp<mlir::emitpybytecode::FunctionCallWithKeywordsOp>(
						op,
						op.getOutput().getType(),
						callee,
						args,
						op.getKeywords(),
						op.getKwargs());
				} else {
					rewriter.replaceOpWithNewOp<mlir::emitpybytecode::FunctionCallOp>(
						op, op.getOutput().getType(), callee, args);
				}

				return success();
			}
		};

		struct ConditionalBranchOpLowering : public mlir::OpRewritePattern<mlir::cf::CondBranchOp>
		{
			using OpRewritePattern<mlir::cf::CondBranchOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::cf::CondBranchOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto cond = op.getCondition();
				ASSERT(mlir::isa<mlir::py::CastToBoolOp>(cond.getDefiningOp()));
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::JumpIfFalse>(op,
					mlir::cast<mlir::py::CastToBoolOp>(cond.getDefiningOp()).getValue(),
					op.getTrueDest(),
					op.getTrueDestOperands(),
					op.getFalseDest(),
					op.getFalseDestOperands());
				return success();
			}
		};

		struct CondBranchSubclassOpLowering
			: public mlir::OpRewritePattern<mlir::py::CondBranchSubclassOp>
		{
			using OpRewritePattern<mlir::py::CondBranchSubclassOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::CondBranchSubclassOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::JumpIfNotException>(op,
					op.getObjectType(),
					op.getTrueDestOperands(),
					op.getFalseDestOperands(),
					op.getTrueDest(),
					op.getFalseDest());

				return success();
			}
		};

		struct CompareOpLowering : public mlir::OpRewritePattern<mlir::py::Compare>
		{
			using OpRewritePattern<mlir::py::Compare>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::Compare op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto lhs = op.getLhs();
				auto rhs = op.getRhs();
				auto op_type = mlir::IntegerAttr::get(
					rewriter.getIntegerType(8, false), static_cast<uint8_t>(op.getPredicate()));

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::Compare>(
					op, op.getOutput().getType(), lhs, rhs, op_type);

				return success();
			}
		};

		struct InplaceOpLowering : public mlir::OpRewritePattern<InplaceOp>
		{
			using OpRewritePattern<InplaceOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(InplaceOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto kind = [&op]() {
					switch (op.getKind()) {
					case py::InplaceOpKind::add: {
						return BinaryOperation::Operation::PLUS;
					} break;
					case py::InplaceOpKind::sub: {
						return BinaryOperation::Operation::MINUS;
					} break;
					case py::InplaceOpKind::mod: {
						return BinaryOperation::Operation::MODULO;
					} break;
					case py::InplaceOpKind::mul: {
						return BinaryOperation::Operation::MULTIPLY;
					} break;
					case py::InplaceOpKind::exp: {
						return BinaryOperation::Operation::EXP;
					} break;
					case py::InplaceOpKind::div: {
						return BinaryOperation::Operation::SLASH;
					} break;
					case py::InplaceOpKind::fldiv: {
						return BinaryOperation::Operation::FLOORDIV;
					} break;
					case py::InplaceOpKind::lshift: {
						return BinaryOperation::Operation::LEFTSHIFT;
					} break;
					case py::InplaceOpKind::rshift: {
						return BinaryOperation::Operation::RIGHTSHIFT;
					} break;
					case py::InplaceOpKind::and_: {
						return BinaryOperation::Operation::AND;
					} break;
					case py::InplaceOpKind::or_: {
						return BinaryOperation::Operation::OR;
					} break;
					case py::InplaceOpKind::xor_: {
						return BinaryOperation::Operation::XOR;
					} break;
					case py::InplaceOpKind::mmul: {
						return BinaryOperation::Operation::MATMUL;
					} break;
					}
					ASSERT_NOT_REACHED();
				}();
				auto dst = op.getDst();
				auto src = op.getSrc();
				auto op_type = mlir::IntegerAttr::get(
					rewriter.getIntegerType(8, false), static_cast<uint8_t>(kind));

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::InplaceOp>(
					op, op.getResult().getType(), dst, src, op_type);

				return success();
			}
		};

		template<typename BinaryOpType, BinaryOperation::Operation OperationEnumType>
		struct BinaryOpLowering : public mlir::OpRewritePattern<BinaryOpType>
		{
			using OpRewritePattern<BinaryOpType>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(BinaryOpType op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto lhs = op.getLhs();
				auto rhs = op.getRhs();
				auto op_type = mlir::IntegerAttr::get(
					rewriter.getIntegerType(8, false), static_cast<uint8_t>(OperationEnumType));

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BinaryOp>(
					op, op.getOutput().getType(), lhs, rhs, op_type);

				return success();
			}
		};

#define BINARY_OP_LOWERING(OPNAME, BINARYOP_ENUM) \
	using OPNAME##Lowering = BinaryOpLowering<py::OPNAME, BinaryOperation::Operation::BINARYOP_ENUM>

		BINARY_OP_LOWERING(BinaryAddOp, PLUS);
		BINARY_OP_LOWERING(BinarySubtractOp, MINUS);
		BINARY_OP_LOWERING(BinaryModuloOp, MODULO);
		BINARY_OP_LOWERING(BinaryMultiplyOp, MULTIPLY);
		BINARY_OP_LOWERING(BinaryExpOp, EXP);
		BINARY_OP_LOWERING(BinaryDivOp, SLASH);
		BINARY_OP_LOWERING(BinaryFloorDivOp, FLOORDIV);
		BINARY_OP_LOWERING(BinaryMatMulOp, MATMUL);
		BINARY_OP_LOWERING(LeftShiftOp, LEFTSHIFT);
		BINARY_OP_LOWERING(RightShiftOp, RIGHTSHIFT);
		BINARY_OP_LOWERING(LogicalAndOp, AND);
		BINARY_OP_LOWERING(LogicalOrOp, OR);
		BINARY_OP_LOWERING(LogicalXorOp, XOR);

#undef BINARY_OP_LOWERING

		struct LoadAssertionErrorOpLowering
			: public mlir::OpRewritePattern<mlir::py::LoadAssertionError>
		{
			using OpRewritePattern<mlir::py::LoadAssertionError>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::LoadAssertionError op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::LoadAssertionError>(
					op, op.getOutput().getType());

				return success();
			}
		};

		struct PositiveOpLowering : public mlir::OpRewritePattern<mlir::py::PositiveOp>
		{
			using OpRewritePattern<mlir::py::PositiveOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::PositiveOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::UnaryOp>(op,
					op.getOutput().getType(),
					op.getInput(),
					static_cast<uint8_t>(Unary::Operation::POSITIVE));

				return success();
			}
		};

		struct NegativeOpLowering : public mlir::OpRewritePattern<mlir::py::NegativeOp>
		{
			using OpRewritePattern<mlir::py::NegativeOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::NegativeOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::UnaryOp>(op,
					op.getOutput().getType(),
					op.getInput(),
					static_cast<uint8_t>(Unary::Operation::NEGATIVE));

				return success();
			}
		};

		struct InvertOpLowering : public mlir::OpRewritePattern<mlir::py::InvertOp>
		{
			using OpRewritePattern<mlir::py::InvertOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::InvertOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::UnaryOp>(op,
					op.getOutput().getType(),
					op.getInput(),
					static_cast<uint8_t>(Unary::Operation::INVERT));

				return success();
			}
		};

		struct NotOpLowering : public mlir::OpRewritePattern<mlir::py::NotOp>
		{
			using OpRewritePattern<mlir::py::NotOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::NotOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::UnaryOp>(op,
					op.getOutput().getType(),
					op.getInput(),
					static_cast<uint8_t>(Unary::Operation::NOT));

				return success();
			}
		};

		struct BuildDictOpLowering : public mlir::OpRewritePattern<mlir::py::BuildDictOp>
		{
			using OpRewritePattern<mlir::py::BuildDictOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::BuildDictOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				const auto &requires_expansion = op.getRequiresExpansion();
				if (std::any_of(requires_expansion.begin(),
						requires_expansion.end(),
						[](const auto &el) { return el == 1; })) {
					std::optional<mlir::Value> result;
					std::vector<mlir::Value> keys;
					std::vector<mlir::Value> values;

					for (auto [key, value, to_expand] :
						llvm::zip(op.getKeys(), op.getValues(), op.getRequiresExpansion())) {
						if (to_expand) {
							if (!result.has_value()) {
								result = rewriter.create<mlir::emitpybytecode::BuildDict>(
									op.getLoc(), op.getOutput().getType(), keys, values);
								keys.clear();
								values.clear();
							}
							rewriter.create<mlir::emitpybytecode::DictUpdate>(
								op.getLoc(), *result, value);
						} else {
							if (!result.has_value()) {
								keys.push_back(key);
								values.push_back(value);
							} else {
								ASSERT(keys.empty());
								ASSERT(values.empty());
								rewriter.create<mlir::emitpybytecode::DictAdd>(
									op.getLoc(), *result, key, value);
							}
						}
					}

					ASSERT(result.has_value());
					ASSERT(keys.empty());
					ASSERT(values.empty());

					rewriter.replaceOp(op, { *result });
				} else {
					rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BuildDict>(
						op, op.getOutput().getType(), op.getKeys(), op.getValues());
				}

				return success();
			}
		};

		struct BuildListOpLowering : public mlir::OpRewritePattern<mlir::py::BuildListOp>
		{
			using OpRewritePattern<mlir::py::BuildListOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::BuildListOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				const auto &requires_expansion = op.getRequiresExpansion();
				if (std::any_of(requires_expansion.begin(),
						requires_expansion.end(),
						[](const auto &el) { return el == 1; })) {
					auto list = rewriter.create<mlir::emitpybytecode::BuildList>(
						op.getLoc(), op.getOutput().getType(), ValueRange{});
					for (auto [el, expand] : llvm::zip(op.getElements(), requires_expansion)) {
						if (expand) {
							rewriter.create<mlir::emitpybytecode::ListExtend>(
								op.getLoc(), list, el);
						} else {
							rewriter.create<mlir::emitpybytecode::ListAppend>(
								op.getLoc(), list, el);
						}
					}
					rewriter.replaceOp(op, list);
				} else {
					rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BuildList>(
						op, op.getOutput().getType(), op.getElements());
				}

				return success();
			}
		};

		struct ListAppendOpLowering : public mlir::OpRewritePattern<mlir::py::ListAppendOp>
		{
			using OpRewritePattern<mlir::py::ListAppendOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::ListAppendOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ListAppend>(
					op, op.getList(), op.getValue());

				return success();
			}
		};

		struct DictAddOpLowering : public mlir::OpRewritePattern<mlir::py::DictAddOp>
		{
			using OpRewritePattern<mlir::py::DictAddOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::DictAddOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::DictAdd>(
					op, op.getDict(), op.getKey(), op.getValue());

				return success();
			}
		};

		struct BuildTupleOpLowering : public mlir::OpRewritePattern<mlir::py::BuildTupleOp>
		{
			using OpRewritePattern<mlir::py::BuildTupleOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::BuildTupleOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				const auto &requires_expansion = op.getRequiresExpansion();
				if (std::any_of(requires_expansion.begin(),
						requires_expansion.end(),
						[](const auto &el) { return el == 1; })) {
					auto list = rewriter.create<mlir::emitpybytecode::BuildList>(
						op.getLoc(), op.getOutput().getType(), ValueRange{});
					for (auto [el, expand] : llvm::zip(op.getElements(), requires_expansion)) {
						if (expand) {
							rewriter.create<mlir::emitpybytecode::ListExtend>(
								op.getLoc(), list, el);
						} else {
							rewriter.create<mlir::emitpybytecode::ListAppend>(
								op.getLoc(), list, el);
						}
					}
					rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ListToTuple>(
						op, op.getOutput().getType(), list);
				} else {
					rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BuildTuple>(
						op, op.getOutput().getType(), op.getElements());
				}

				return success();
			}
		};

		struct BuildSetOpLowering : public mlir::OpRewritePattern<mlir::py::BuildSetOp>
		{
			using OpRewritePattern<mlir::py::BuildSetOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::BuildSetOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				const auto &requires_expansion = op.getRequiresExpansion();
				if (std::any_of(requires_expansion.begin(),
						requires_expansion.end(),
						[](const auto &el) { return el == 1; })) {
					std::vector<mlir::Value> elements;
					std::optional<mlir::Value> set;
					for (auto [el, expand] : llvm::zip(op.getElements(), requires_expansion)) {
						if (expand) {
							if (!set.has_value()) {
								set = rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BuildSet>(
									op, op.getOutput().getType(), elements);
							} else {
								for (auto el : elements) {
									rewriter.create<mlir::emitpybytecode::SetAdd>(
										op.getLoc(), *set, el);
								}
							}
							elements.clear();
							rewriter.create<mlir::emitpybytecode::SetUpdate>(op.getLoc(), *set, el);
						} else {
							elements.push_back(el);
						}
					}
					ASSERT(set.has_value());
					for (auto el : elements) {
						rewriter.create<mlir::emitpybytecode::SetAdd>(op.getLoc(), *set, el);
					}
				} else {
					rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BuildSet>(
						op, op.getOutput().getType(), op.getElements());
				}

				return success();
			}
		};

		struct SetAddOpLowering : public mlir::OpRewritePattern<mlir::py::SetAddOp>
		{
			using OpRewritePattern<mlir::py::SetAddOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::SetAddOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::SetAdd>(
					op, op.getSet(), op.getValue());

				return success();
			}
		};

		struct BuildStringOpLowering : public mlir::OpRewritePattern<mlir::py::BuildStringOp>
		{
			using OpRewritePattern<mlir::py::BuildStringOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::BuildStringOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BuildString>(
					op, op.getOutput().getType(), op.getElements());

				return success();
			}
		};

		struct FormatValueOpLowering : public mlir::OpRewritePattern<mlir::py::FormatValueOp>
		{
			using OpRewritePattern<mlir::py::FormatValueOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::FormatValueOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::FormatValue>(op,
					op.getOutput().getType(),
					op.getValue(),
					static_cast<uint8_t>(op.getConversion()));

				return success();
			}
		};

		struct LoadAttributeOpLowering : public mlir::OpRewritePattern<mlir::py::LoadAttributeOp>
		{
			using OpRewritePattern<mlir::py::LoadAttributeOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::LoadAttributeOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::LoadAttribute>(
					op, op.getOutput().getType(), op.getSelf(), op.getAttr());

				return success();
			}
		};

		struct LoadMethodOpLowering : public mlir::OpRewritePattern<mlir::py::LoadMethodOp>
		{
			using OpRewritePattern<mlir::py::LoadMethodOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::LoadMethodOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto parent_fn = op->getParentOfType<mlir::func::FuncOp>();
				add_identifier(parent_fn, op.getMethodName(), rewriter);

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::LoadMethod>(
					op, op.getMethod().getType(), op.getSelf(), op.getMethodName());

				return success();
			}
		};

		struct BinarySubscriptOpLowering
			: public mlir::OpRewritePattern<mlir::py::BinarySubscriptOp>
		{
			using OpRewritePattern<mlir::py::BinarySubscriptOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::BinarySubscriptOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BinarySubscript>(
					op, op.getOutput().getType(), op.getSelf(), op.getSubscript());

				return success();
			}
		};

		struct StoreSubscriptOpLowering : public mlir::OpRewritePattern<mlir::py::StoreSubscriptOp>
		{
			using OpRewritePattern<mlir::py::StoreSubscriptOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::StoreSubscriptOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::StoreSubscript>(
					op, op.getSelf(), op.getSubscript(), op.getValue());

				return success();
			}
		};

		struct DeleteSubscriptOpLowering
			: public mlir::OpRewritePattern<mlir::py::DeleteSubscriptOp>
		{
			using OpRewritePattern<mlir::py::DeleteSubscriptOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::DeleteSubscriptOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::DeleteSubscript>(
					op, op.getSelf(), op.getSubscript());

				return success();
			}
		};

		struct StoreAttributeOpLowering : public mlir::OpRewritePattern<mlir::py::StoreAttributeOp>
		{
			using OpRewritePattern<mlir::py::StoreAttributeOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::StoreAttributeOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto parent_fn = op->getParentOfType<mlir::func::FuncOp>();
				add_identifier(parent_fn, op.getAttribute(), rewriter);

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::StoreAttribute>(
					op, op.getSelf(), op.getAttribute(), op.getValue());

				return success();
			}
		};

		struct BuildSliceOpLowering : public mlir::OpRewritePattern<mlir::py::BuildSliceOp>
		{
			using OpRewritePattern<mlir::py::BuildSliceOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::BuildSliceOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::BuildSlice>(
					op, op.getSlice().getType(), op.getLower(), op.getUpper(), op.getStep());

				return success();
			}
		};

		struct MakeFunctionOpLowering : public mlir::OpRewritePattern<mlir::py::MakeFunctionOp>
		{
			using OpRewritePattern<mlir::py::MakeFunctionOp>::OpRewritePattern;
			mlir::LogicalResult matchAndRewrite(mlir::py::MakeFunctionOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto module = op->getParentOfType<mlir::ModuleOp>();
				auto function_definition = module.lookupSymbol(op.getFunctionName());
				ASSERT(function_definition);
				ASSERT(mlir::isa<mlir::func::FuncOp>(*function_definition));

				auto sym_name = rewriter.create<mlir::emitpybytecode::ConstantOp>(op.getLoc(),
					mlir::py::PyObjectType::get(rewriter.getContext()),
					rewriter.getStringAttr(op.getFunctionName()));

				auto captures_tuple = [&]() -> mlir::Value {
					if (op.getCaptures().empty()) { return nullptr; }
					std::vector<mlir::Value> captures_vec;
					for (auto name : op.getCaptures().getValues<mlir::StringRef>()) {
						captures_vec.push_back(rewriter.create<mlir::emitpybytecode::LoadClosureOp>(
							op.getLoc(), mlir::py::PyObjectType::get(getContext()), name));
					}
					return rewriter.create<mlir::emitpybytecode::BuildTuple>(
						op.getLoc(), mlir::py::PyObjectType::get(getContext()), captures_vec);
				}();
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::MakeFunction>(op,
					mlir::py::PyObjectType::get(rewriter.getContext()),
					sym_name,
					op.getDefaults(),
					op.getKwDefaults(),
					captures_tuple);

				return success();
			}
		};

		struct FuncOpLowering : public mlir::OpRewritePattern<mlir::func::FuncOp>
		{
			using OpRewritePattern<mlir::func::FuncOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::func::FuncOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				if (op.isPrivate()) { return success(); }
				// if (op.isPrivate() && op.getSymName() == "__hidden_init__") { return success(); }
				populate_arguments(op, rewriter);

				return success();
			}

			void populate_arguments(mlir::func::FuncOp &op, mlir::OpBuilder &builder) const
			{
				for (size_t i = 0; i < op.getNumArguments(); ++i) {
					auto arg_name = op.getArgAttr(i, "llvm.name");
					ASSERT(arg_name);
					add_local(op, arg_name.cast<mlir::StringAttr>().getValue(), builder);
				}
			}

			void add_local(mlir::func::FuncOp &fn,
				mlir::StringRef identifier,
				mlir::OpBuilder &builder) const
			{
				if (fn->hasAttr("locals")) {
					auto names = fn->getAttr("locals");
					std::vector<StringRef> names_vec;
					auto arr = names.cast<mlir::ArrayAttr>().getValue();
					if (std::find_if(arr.begin(),
							arr.end(),
							[identifier](mlir::Attribute attr) {
								return attr.cast<mlir::StringAttr>().getValue() == identifier;
							})
						!= arr.end()) {
						return;
					}
					std::transform(arr.begin(),
						arr.end(),
						std::back_inserter(names_vec),
						[](mlir::Attribute attr) {
							return attr.cast<mlir::StringAttr>().getValue();
						});
					names_vec.emplace_back(identifier);
					fn->setAttr("locals", builder.getStrArrayAttr(names_vec));
				} else {
					fn->setAttr("locals", builder.getStrArrayAttr({ identifier }));
				}
			}
		};

		struct ClassDefinitionOpLowering
			: public mlir::OpRewritePattern<mlir::py::ClassDefinitionOp>
		{
			using OpRewritePattern<mlir::py::ClassDefinitionOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::ClassDefinitionOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto module = op->getParentOfType<mlir::ModuleOp>();
				rewriter.setInsertionPointToEnd(module.getBody());

				auto func_type = rewriter.getFunctionType(mlir::TypeRange{},
					mlir::TypeRange{ mlir::py::PyObjectType::get(rewriter.getContext()) });
				auto class_fn_definition = rewriter.create<mlir::func::FuncOp>(op.getLoc(),
					op.getMangledName(),
					func_type,
					mlir::ArrayRef<mlir::NamedAttribute>{},
					mlir::ArrayRef<mlir::DictionaryAttr>{});

				class_fn_definition->setAttr("is_class", rewriter.getBoolAttr(true));

				if (auto cellvars = op->getAttrOfType<mlir::ArrayAttr>("cellvars")) {
					auto cell_names = cellvars.getValue();
					if (std::find_if(cell_names.begin(),
							cell_names.end(),
							[](mlir::Attribute name) {
								return name.cast<mlir::StringAttr>() == "__class__";
							})
						!= cell_names.end()) {

						mlir::Operation *return_op_;
						op.getBody().walk([&return_op_](mlir::Operation *child_op) {
							if (mlir::isa<mlir::func::FuncOp>(child_op)) {
								return WalkResult::skip();
							}
							if (mlir::isa<mlir::func::ReturnOp>(child_op)) {
								return_op_ = child_op;
								return WalkResult::interrupt();
							}
							return WalkResult::advance();
						});
						ASSERT(return_op_);
						auto return_op = mlir::cast<mlir::func::ReturnOp>(return_op_);
						ASSERT(return_op.getOperands().size() == 1);
						ASSERT(return_op.getOperand(0).getDefiningOp());
						rewriter.setInsertionPoint(return_op.getOperand(0).getDefiningOp());
						rewriter.replaceOpWithNewOp<mlir::emitpybytecode::LoadClosureOp>(
							return_op.getOperand(0).getDefiningOp(),
							mlir::py::PyObjectType::get(getContext()),
							mlir::StringRef{ "__class__" });
					}
				}

				auto attr = class_fn_definition->getAttrs().vec();
				attr.insert(attr.end(), op->getAttrs().begin(), op->getAttrs().end());
				class_fn_definition->setAttrs(attr);

				auto *end = class_fn_definition.addEntryBlock();
				rewriter.setInsertionPointToStart(end);
				rewriter.inlineRegionBefore(op.getBody(), &class_fn_definition.getBody().front());
				rewriter.eraseBlock(end);

				rewriter.setInsertionPoint(op);
				auto class_name = rewriter.create<mlir::emitpybytecode::ConstantOp>(op.getLoc(),
					mlir::py::PyObjectType::get(rewriter.getContext()),
					rewriter.getStringAttr(op.getMangledName()));

				auto captures_tuple = [&]() -> mlir::Value {
					if (op.getCaptures().empty()) { return {}; }
					std::vector<mlir::Value> captures_vec;
					for (auto name : op.getCaptures().getValues<mlir::StringRef>()) {
						captures_vec.push_back(rewriter.create<mlir::emitpybytecode::LoadClosureOp>(
							op.getLoc(), mlir::py::PyObjectType::get(getContext()), name));
					}
					return rewriter.create<mlir::emitpybytecode::BuildTuple>(
						op.getLoc(), mlir::py::PyObjectType::get(getContext()), captures_vec);
				}();

				auto class_fn = rewriter.create<mlir::emitpybytecode::MakeFunction>(op.getLoc(),
					mlir::py::PyObjectType::get(rewriter.getContext()),
					class_name,
					mlir::ValueRange{},
					mlir::ValueRange{},
					captures_tuple);

				auto class_builder = rewriter.create<mlir::emitpybytecode::LoadBuildClass>(
					op.getLoc(), mlir::py::PyObjectType::get(rewriter.getContext()));
				std::vector<mlir::Value> args{ class_fn, class_name };
				args.insert(args.end(), op.getBases().begin(), op.getBases().end());
				rewriter.replaceOpWithNewOp<py::FunctionCallOp>(op,
					op.getOutput().getType(),
					class_builder,
					args,
					op.getKeywords(),
					op.getKwargs(),
					false,
					false);

				return success();
			}
		};

		struct ForLoopOpLowering : public mlir::OpRewritePattern<mlir::py::ForLoopOp>
		{
			using OpRewritePattern<mlir::py::ForLoopOp>::OpRewritePattern;

			std::vector<mlir::Value> getIterators(mlir::py::ForLoopOp op,
				mlir::emitpybytecode::GetIter current_iterator) const
			{
				std::vector<mlir::Value> iterators;

				iterators.push_back(current_iterator);

				auto parent = op->getParentOfType<mlir::py::ForLoopOp>();
				while (parent) {
					auto iterable = parent.getIterable();
					ASSERT(!iterable.getUsers().empty());
					auto iterator = *iterable.getUsers().begin();
					ASSERT(mlir::isa<mlir::emitpybytecode::GetIter>(*iterator));
					iterators.insert(
						iterators.end() - 1, mlir::cast<mlir::emitpybytecode::GetIter>(*iterator));
					parent = parent->getParentOfType<mlir::py::ForLoopOp>();
				}

				return iterators;
			}

			mlir::LogicalResult matchAndRewrite(mlir::py::ForLoopOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto *initBlock = rewriter.getInsertionBlock();
				auto initPos = rewriter.getInsertionPoint();

				auto *endBlock = rewriter.splitBlock(initBlock, initPos);

				auto iterable = op.getIterable();
				rewriter.setInsertionPointToEnd(initBlock);
				auto iterator = rewriter.create<mlir::emitpybytecode::GetIter>(
					op.getStep().getLoc(), iterable.getType(), iterable);

				// advance iterator
				auto iterator_next_block = rewriter.createBlock(endBlock);
				// iterator_next_block->addArgument(iterator.getType(), op.getStep().getLoc());
				rewriter.setInsertionPointToEnd(initBlock);
				const auto &iterators = getIterators(op, iterator);
				rewriter.create<mlir::cf::BranchOp>(op.getStep().getLoc(), iterator_next_block);

				rewriter.setInsertionPointToStart(iterator_next_block);

				rewriter.create<mlir::emitpybytecode::ForIter>(op.getStep().getLoc(),
					// iterator_next_block->getArgument(0),
					iterators.front(),
					&op.getStep().front(),
					op.getOrelse().empty() ? endBlock : &op.getOrelse().front());

				ASSERT(!op.getStep().empty())
				auto *iterator_exit_block = &op.getStep().back();
				ASSERT(iterator_exit_block->getTerminator());
				ASSERT(mlir::isa<mlir::py::ControlFlowYield>(iterator_exit_block->getTerminator()));

				rewriter.setInsertionPointToEnd(iterator_exit_block);
				rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
					iterator_exit_block->getTerminator(), &op.getBody().front() /*, iterators*/);

				auto *for_iter_block = rewriter.createBlock(&op.getBody());
				// for (const auto &it : iterators) {
				// 	for_iter_block->addArgument(it.getType(), op.getStep().getLoc());
				// }
				rewriter.create<mlir::emitpybytecode::ForIter>(op.getStep().getLoc(),
					iterators.front(),
					&op.getStep().front(),
					op.getOrelse().empty() ? endBlock : &op.getOrelse().front());

				rewriter.inlineRegionBefore(
					op.getStep(), *op->getParentRegion(), endBlock->getIterator());

				// for (const auto &it : iterators) {
				// 	op.getBody().addArgument(it.getType(), op.getStep().getLoc());
				// }

				op.getBody().walk<WalkOrder::PreOrder>([&](mlir::Operation *operation) {
					if (mlir::isa<mlir::py::ForLoopOp, mlir::py::WhileOp>(operation)) {
						return WalkResult::skip();
					}
					if (auto yield_op = mlir::dyn_cast<mlir::py::ControlFlowYield>(operation)) {
						static_assert(mlir::py::ControlFlowYield::hasTrait<mlir::OpTrait::
								HasParent<TryOp, ForLoopOp, WithOp, WhileOp, TryHandlerScope>::
									Impl>());
						if (!yield_op.getKind().has_value()
							&& mlir::isa<TryOp, WithOp, TryHandlerScope>(yield_op->getParentOp())) {
							return WalkResult::advance();
						}

						rewriter.setInsertionPoint(yield_op);
						if (!yield_op.getKind().has_value()
							|| yield_op.getKind().value() == py::LoopOpKind::continue_) {
							rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
								yield_op, for_iter_block);
						} else if (yield_op.getKind().value() == py::LoopOpKind::break_) {
							rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yield_op, endBlock);
						}
					}
					return WalkResult::advance();
				});
				ASSERT(!op.getBody().empty());
				auto *body_exit_block = &op.getBody().back();
				ASSERT(body_exit_block->getTerminator());
				rewriter.inlineRegionBefore(
					op.getBody(), *op->getParentRegion(), endBlock->getIterator());

				if (!op.getOrelse().empty()) {
					auto *orelse_exit_block = &op.getOrelse().back();
					ASSERT(orelse_exit_block->getTerminator());
					if (mlir::isa<mlir::py::ControlFlowYield>(orelse_exit_block->getTerminator())) {
						rewriter.setInsertionPointToEnd(orelse_exit_block);
						rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
							orelse_exit_block->getTerminator(), endBlock);
					}
				}
				rewriter.inlineRegionBefore(
					op.getOrelse(), *op->getParentRegion(), endBlock->getIterator());

				rewriter.eraseOp(op);
				return success();
			}
		};

		struct WhileOpLowering : public mlir::OpRewritePattern<mlir::py::WhileOp>
		{
			using OpRewritePattern<mlir::py::WhileOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::WhileOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto *initBlock = rewriter.getInsertionBlock();
				auto initPos = rewriter.getInsertionPoint();

				auto *endBlock = rewriter.splitBlock(initBlock, initPos);

				auto &condition = op.getCondition();
				auto &condition_start = condition.getBlocks().front();
				ASSERT(!condition.getBlocks().empty());
				ASSERT(condition.back().getTerminator());

				auto condition_op =
					mlir::cast<mlir::py::ConditionOp>(condition.back().getTerminator());
				ASSERT(condition_op);

				rewriter.setInsertionPointToEnd(initBlock);
				rewriter.create<mlir::cf::BranchOp>(condition_op.getLoc(), &condition_start);

				if (mlir::isa<mlir::BlockArgument>(condition_op.getCond())) {
					rewriter.setInsertionPointToStart(condition_op.getCond().getParentBlock());
				} else {
					rewriter.setInsertionPointAfter(condition_op.getCond().getDefiningOp());
				}
				auto should_jump = rewriter.create<mlir::py::CastToBoolOp>(
					condition_op.getLoc(), rewriter.getI1Type(), condition_op.getCond());
				ASSERT(!op.getBody().empty());
				rewriter.create<mlir::cf::CondBranchOp>(condition_op.getLoc(),
					should_jump,
					&op.getBody().front(),
					op.getOrelse().empty() ? endBlock : &op.getOrelse().front());
				rewriter.eraseOp(condition_op);
				rewriter.inlineRegionBefore(condition, endBlock);

				op.getBody().walk<WalkOrder::PreOrder>([&](mlir::Operation *operation) {
					if (mlir::isa<mlir::py::ForLoopOp, mlir::py::WhileOp>(operation)) {
						return WalkResult::skip();
					}
					if (auto yield_op = mlir::dyn_cast<mlir::py::ControlFlowYield>(operation)) {
						static_assert(mlir::py::ControlFlowYield::hasTrait<mlir::OpTrait::
								HasParent<TryOp, ForLoopOp, WithOp, WhileOp, TryHandlerScope>::
									Impl>());
						if (!yield_op.getKind().has_value()
							&& mlir::isa<TryOp, WithOp, TryHandlerScope>(yield_op->getParentOp())) {
							return WalkResult::advance();
						}
						rewriter.setInsertionPoint(yield_op);
						if (!yield_op.getKind().has_value()
							|| yield_op.getKind().value() == py::LoopOpKind::continue_) {
							rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
								yield_op, &condition_start);
						} else if (yield_op.getKind().value() == py::LoopOpKind::break_) {
							rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(yield_op, endBlock);
						}
					}
					return WalkResult::advance();
				});
				rewriter.inlineRegionBefore(op.getBody(), endBlock);

				if (!op.getOrelse().empty()) {
					auto *orelse_exit_block = &op.getOrelse().back();
					ASSERT(orelse_exit_block->getTerminator());
					ASSERT(
						mlir::isa<mlir::py::ControlFlowYield>(orelse_exit_block->getTerminator()));
					rewriter.setInsertionPointToEnd(orelse_exit_block);
					rewriter.replaceOpWithNewOp<mlir::cf::BranchOp>(
						orelse_exit_block->getTerminator(), endBlock);
				}
				rewriter.inlineRegionBefore(op.getOrelse(), endBlock);

				rewriter.eraseOp(op);

				return success();
			}
		};

		struct TryOpLowering : public mlir::OpRewritePattern<mlir::py::TryOp>
		{
			using OpRewritePattern<mlir::py::TryOp>::OpRewritePattern;

			template<typename FnT>
			void replace_controlflow_yield(mlir::Region &region, FnT &&callback) const
			{
				if (region.empty()) { return; }
				region.walk<WalkOrder::PreOrder>([callback](mlir::Operation *childOp) {
					static_assert(mlir::py::ControlFlowYield::hasTrait<mlir::OpTrait::
							HasParent<TryOp, ForLoopOp, WithOp, WhileOp, TryHandlerScope>::Impl>());
					if (mlir::isa<mlir::py::TryOp>(childOp)
						|| mlir::isa<mlir::py::ForLoopOp>(childOp)
						|| mlir::isa<mlir::py::WhileOp>(childOp)
						|| mlir::isa<mlir::py::WithOp>(childOp)
						|| mlir::isa<mlir::py::TryHandlerScope>(childOp)) {
						return WalkResult::skip();
					}
					if (mlir::isa<mlir::py::ControlFlowYield>(childOp)
						&& !mlir::cast<mlir::py::ControlFlowYield>(childOp).getKind().has_value()) {
						callback(childOp);
					}
					return WalkResult::advance();
				});
			}

			mlir::LogicalResult matchAndRewrite(mlir::py::TryOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto *initBlock = rewriter.getInsertionBlock();
				auto initPos = rewriter.getInsertionPoint();

				auto *endBlock = rewriter.splitBlock(initBlock, initPos);

				auto *body_start = &op.getBody().front();

				replace_controlflow_yield(
					op.getBody(), [&rewriter, &op, endBlock](mlir::Operation *childOp) {
						auto *current = childOp->getBlock();
						auto *next = rewriter.splitBlock(current, childOp->getIterator());
						rewriter.setInsertionPointToEnd(current);
						rewriter.create<mlir::emitpybytecode::LeaveExceptionHandle>(
							childOp->getLoc());
						if (op.getHandlers().empty()) {
							rewriter.create<mlir::cf::BranchOp>(
								childOp->getLoc(), &op.getFinally().front());
						} else if (!op.getOrelse().empty()) {
							rewriter.create<mlir::cf::BranchOp>(
								childOp->getLoc(), &op.getOrelse().front());
						} else if (!op.getFinally().empty()) {
							rewriter.create<mlir::cf::BranchOp>(
								childOp->getLoc(), &op.getFinally().front());
						} else {
							rewriter.create<mlir::cf::BranchOp>(childOp->getLoc(), endBlock);
						}
						rewriter.eraseOp(childOp);
						rewriter.eraseBlock(next);
					});
				rewriter.inlineRegionBefore(op.getBody(), endBlock);

				std::optional<mlir::IRMapping> finally_mapping;
				if (!op.getFinally().empty()) {
					finally_mapping = mlir::IRMapping{};

					rewriter.cloneRegionBefore(op.getFinally(),
						*endBlock->getParent(),
						endBlock->getIterator(),
						*finally_mapping);

					replace_controlflow_yield(op.getFinally(),
						[&rewriter, &op, &finally_mapping, endBlock](mlir::Operation *childOp) {
							{
								auto *current = childOp->getBlock();
								auto *next = rewriter.splitBlock(current, childOp->getIterator());
								rewriter.setInsertionPointToEnd(current);
								rewriter.create<mlir::cf::BranchOp>(childOp->getLoc(), endBlock);
								rewriter.eraseOp(childOp);
								rewriter.eraseBlock(next);
							}

							childOp = finally_mapping->lookup(childOp);
							{
								auto *current = childOp->getBlock();
								auto *next = rewriter.splitBlock(current, childOp->getIterator());
								rewriter.setInsertionPointToEnd(current);
								rewriter.create<mlir::py::RaiseOp>(childOp->getLoc());
								rewriter.eraseOp(childOp);
								rewriter.eraseBlock(next);
							}
						});
				}

				rewriter.setInsertionPointToEnd(initBlock);
				if (!op.getHandlers().empty()) {
					auto &handler = op.getHandlers().front();
					ASSERT(handler.getBlocks().size() == 1);
					auto handler_scope =
						mlir::cast<mlir::py::TryHandlerScope>(handler.front().getTerminator());
					ASSERT(handler_scope);
					rewriter.create<mlir::emitpybytecode::SetupExceptionHandle>(op.getLoc(),
						body_start,
						handler_scope.getCond().empty() ? &handler_scope.getHandler().front()
														: &handler_scope.getCond().front());
				} else {
					rewriter.create<mlir::emitpybytecode::SetupExceptionHandle>(
						op.getLoc(), body_start, finally_mapping->lookup(&op.getFinally().front()));
				}

				if (!op.getHandlers().empty()) {
					for (auto e : llvm::enumerate(op.getHandlers().drop_back())) {
						auto &handler = e.value();
						auto idx = e.index();

						ASSERT(handler.getBlocks().size() == 1);
						auto handler_scope =
							mlir::cast<mlir::py::TryHandlerScope>(handler.front().getTerminator());
						ASSERT(handler_scope);

						if (!handler_scope.getCond().empty()) {
							auto cond = mlir::cast<mlir::py::ConditionOp>(
								handler_scope.getCond().back().getTerminator());
							ASSERT(cond);
							rewriter.setInsertionPoint(cond);
							auto &next_handler = op.getHandlers()[idx + 1];
							ASSERT(next_handler.getBlocks().size() == 1);
							auto next_handler_scope = mlir::cast<mlir::py::TryHandlerScope>(
								next_handler.front().getTerminator());
							ASSERT(next_handler_scope);

							rewriter.replaceOpWithNewOp<mlir::py::CondBranchSubclassOp>(cond,
								cond.getCond(),
								mlir::ValueRange{},
								mlir::ValueRange{},
								next_handler_scope.getCond().empty()
									? &next_handler_scope.getHandler().front()
									: &next_handler_scope.getCond().front(),
								&handler_scope.getHandler().front());
							rewriter.inlineRegionBefore(handler_scope.getCond(), endBlock);
						}
						replace_controlflow_yield(handler_scope.getHandler(),
							[&rewriter, &op, endBlock](mlir::Operation *childOp) {
								auto *current = childOp->getBlock();
								auto *next = rewriter.splitBlock(current, childOp->getIterator());
								rewriter.setInsertionPointToEnd(current);
								rewriter.create<mlir::emitpybytecode::ClearExceptionState>(
									op.getLoc());
								if (!op.getFinally().empty()) {
									rewriter.create<mlir::cf::BranchOp>(
										childOp->getLoc(), &op.getFinally().front());
								} else {
									rewriter.create<mlir::cf::BranchOp>(
										childOp->getLoc(), endBlock);
								}
								rewriter.eraseOp(childOp);
								rewriter.eraseBlock(next);
							});
						rewriter.inlineRegionBefore(handler_scope.getHandler(), endBlock);
					}

					{
						auto &handler = op.getHandlers().back();
						ASSERT(handler.getBlocks().size() == 1);
						auto handler_scope =
							mlir::cast<mlir::py::TryHandlerScope>(handler.front().getTerminator());
						ASSERT(handler_scope);
						if (!handler_scope.getCond().empty()) {
							auto cond = mlir::cast<mlir::py::ConditionOp>(
								handler_scope.getCond().back().getTerminator());
							ASSERT(cond);

							auto *reraise_block = rewriter.createBlock(&handler_scope.getCond());
							rewriter.create<mlir::py::RaiseOp>(cond.getLoc());

							rewriter.setInsertionPoint(cond);
							rewriter.replaceOpWithNewOp<mlir::py::CondBranchSubclassOp>(cond,
								cond.getCond(),
								mlir::ValueRange{},
								mlir::ValueRange{},
								op.getFinally().empty()
									? reraise_block
									: finally_mapping->lookup(&op.getFinally().front()),
								&handler_scope.getHandler().front());

							rewriter.inlineRegionBefore(handler_scope.getCond(), endBlock);
						}

						replace_controlflow_yield(handler_scope.getHandler(),
							[&rewriter, &op, endBlock](mlir::Operation *childOp) {
								auto *current = childOp->getBlock();
								auto *next = rewriter.splitBlock(current, childOp->getIterator());
								rewriter.setInsertionPointToEnd(current);
								rewriter.create<mlir::emitpybytecode::ClearExceptionState>(
									op.getLoc());
								if (!op.getFinally().empty()) {
									rewriter.create<mlir::cf::BranchOp>(
										childOp->getLoc(), &op.getFinally().front());
								} else {
									rewriter.create<mlir::cf::BranchOp>(
										childOp->getLoc(), endBlock);
								}
								rewriter.eraseOp(childOp);
								rewriter.eraseBlock(next);
							});
						rewriter.inlineRegionBefore(handler_scope.getHandler(), endBlock);
					}
				}

				replace_controlflow_yield(
					op.getOrelse(), [&rewriter, &op, endBlock](mlir::Operation *childOp) {
						auto *current = childOp->getBlock();
						auto *next = rewriter.splitBlock(current, childOp->getIterator());
						rewriter.setInsertionPointToEnd(current);
						if (!op.getFinally().empty()) {
							rewriter.create<mlir::cf::BranchOp>(
								childOp->getLoc(), &op.getFinally().front());
						} else {
							rewriter.create<mlir::cf::BranchOp>(childOp->getLoc(), endBlock);
						}
						rewriter.eraseOp(childOp);
						rewriter.eraseBlock(next);
					});
				rewriter.inlineRegionBefore(op.getOrelse(), endBlock);

				rewriter.inlineRegionBefore(op.getFinally(), endBlock);

				rewriter.eraseOp(op);

				return success();
			}
		};

		struct WithOpLowering : public mlir::OpRewritePattern<mlir::py::WithOp>
		{
			using OpRewritePattern<mlir::py::WithOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::WithOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto *initBlock = rewriter.getInsertionBlock();
				auto initPos = rewriter.getInsertionPoint();

				auto *endBlock = rewriter.splitBlock(initBlock, initPos);

				auto *body_start = &op.getBody().front();
				auto *cleanup_block = rewriter.createBlock(endBlock);
				auto *exit_block = rewriter.createBlock(endBlock);

				op.getBody().walk<WalkOrder::PreOrder>([&rewriter, exit_block, cleanup_block](
														   mlir::Operation *childOp) {
					static_assert(mlir::py::ControlFlowYield::hasTrait<mlir::OpTrait::
							HasParent<TryOp, ForLoopOp, WithOp, WhileOp, TryHandlerScope>::Impl>());
					if (mlir::isa<mlir::py::TryOp>(childOp)
						|| mlir::isa<mlir::py::ForLoopOp>(childOp)
						|| mlir::isa<mlir::py::WhileOp>(childOp)
						|| mlir::isa<mlir::py::WithOp>(childOp)
						|| mlir::isa<mlir::py::TryHandlerScope>(childOp)) {
						return WalkResult::skip();
					}
					if (auto op = mlir::dyn_cast<mlir::py::RaiseOp>(childOp)) {
						rewriter.setInsertionPoint(op);
						if (op.getCause()) {
							rewriter.replaceOpWithNewOp<mlir::emitpybytecode::RaiseVarargs>(
								op, op.getException(), op.getCause(), BlockRange{ cleanup_block });
						} else if (op.getException()) {
							rewriter.replaceOpWithNewOp<mlir::emitpybytecode::RaiseVarargs>(
								op, op.getException(), nullptr, BlockRange{ cleanup_block });
						} else {
							rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ReRaiseOp>(
								op, BlockRange{ cleanup_block });
						}
					} else if (auto op = mlir::dyn_cast<mlir::py::ControlFlowYield>(childOp);
							   op && !op.getKind().has_value()) {
						auto *current = op->getBlock();
						auto *next = rewriter.splitBlock(current, op->getIterator());
						rewriter.setInsertionPointToEnd(current);
						rewriter.create<mlir::emitpybytecode::LeaveExceptionHandle>(op->getLoc());
						rewriter.create<mlir::cf::BranchOp>(op->getLoc(), exit_block);
						rewriter.eraseOp(op);
						rewriter.eraseBlock(next);
					}
					return WalkResult::advance();
				});

				rewriter.inlineRegionBefore(op.getBody(), endBlock);

				rewriter.setInsertionPointToStart(cleanup_block);
				for (const auto &item : op.getItems()) {
					auto exit = rewriter.create<mlir::py::LoadMethodOp>(item.getLoc(),
						mlir::py::PyObjectType::get(rewriter.getContext()),
						item,
						"__exit__");

					auto except_result = rewriter.create<mlir::py::WithExceptStartOp>(
						item.getLoc(), mlir::py::PyObjectType::get(rewriter.getContext()), exit);

					auto *reraise_block = rewriter.createBlock(endBlock);
					auto *continue_block = rewriter.createBlock(endBlock);
					rewriter.setInsertionPointAfter(except_result);

					auto cond = rewriter.create<mlir::py::CastToBoolOp>(
						except_result.getLoc(), rewriter.getI1Type(), except_result);
					rewriter.create<mlir::cf::CondBranchOp>(
						cond.getLoc(), cond, continue_block, reraise_block);

					rewriter.setInsertionPointToStart(reraise_block);
					rewriter.create<mlir::emitpybytecode::ReRaiseOp>(item.getLoc(), endBlock);

					// TODO: handle multiple handlers
					rewriter.setInsertionPointToStart(continue_block);
					rewriter.create<mlir::emitpybytecode::ClearExceptionState>(item.getLoc());
					rewriter.create<mlir::cf::BranchOp>(op.getLoc(), endBlock);
				}
				// rewriter.create<mlir::cf::BranchOp>(op.getLoc(), endBlock);

				rewriter.setInsertionPointToStart(exit_block);
				for (const auto &item : op.getItems()) {
					auto exit = rewriter.create<mlir::py::LoadMethodOp>(item.getLoc(),
						mlir::py::PyObjectType::get(rewriter.getContext()),
						item,
						"__exit__");

					auto none = rewriter.create<mlir::py::ConstantOp>(
						item.getLoc(), rewriter.getNoneType());

					rewriter.create<mlir::py::FunctionCallOp>(item.getLoc(),
						mlir::py::PyObjectType::get(rewriter.getContext()),
						exit,
						std::vector<mlir::Value>{ none, none, none },
						mlir::DenseStringElementsAttr::get(
							mlir::VectorType::get(
								{ 0 }, mlir::StringAttr::get(rewriter.getContext()).getType()),
							{}),
						std::vector<mlir::Value>{},
						false,
						false);

					rewriter.create<mlir::py::ClearExceptionStateOp>(item.getLoc());
				}

				rewriter.create<mlir::cf::BranchOp>(op.getLoc(), endBlock);

				rewriter.setInsertionPointToEnd(initBlock);
				rewriter.create<mlir::emitpybytecode::SetupWith>(
					op.getLoc(), body_start, cleanup_block);

				rewriter.eraseOp(op);

				return success();
			}
		};

		struct WithExceptStartOpLowering
			: public mlir::OpRewritePattern<mlir::py::WithExceptStartOp>
		{
			using OpRewritePattern<mlir::py::WithExceptStartOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::WithExceptStartOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::WithExceptStart>(
					op, op.getOutput().getType(), op.getExitMethod());
				return success();
			}
		};

		struct ClearExceptionStateOpLowering
			: public mlir::OpRewritePattern<mlir::py::ClearExceptionStateOp>
		{
			using OpRewritePattern<mlir::py::ClearExceptionStateOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::ClearExceptionStateOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ClearExceptionState>(op);
				return success();
			}
		};

		struct RaiseOpLowering : public mlir::OpRewritePattern<mlir::py::RaiseOp>
		{
			using OpRewritePattern<mlir::py::RaiseOp>::OpRewritePattern;

			/// Find the first parent operation of the given type, or nullptr if there is
			/// no ancestor operation.
			template<typename... ParentTs>
			static mlir::Operation *getParentOfType(mlir::Region *region)
			{
				do {
					if ((... || mlir::isa<ParentTs>(*region->getParentOp())))
						return region->getParentOp();
				} while ((region = region->getParentRegion()));
				return nullptr;
			}

			static mlir::Block *get_handler(mlir::Operation *op, mlir::PatternRewriter &rewriter)
			{
				// find possible catch block in order to not clobber an active result register
				auto *handler_op =
					getParentOfType<mlir::py::TryOp, mlir::py::WithOp, mlir::func::FuncOp>(
						op->getParentRegion());
				ASSERT(handler_op);
				return llvm::TypeSwitch<mlir::Operation *, mlir::Block *>(handler_op)
					.Case([](mlir::py::TryOp op) {
						return op.getHandlers().empty() ? &op.getFinally().front()
														: &op.getHandlers().front().front();
					})
					.Case([](mlir::py::WithOp op) { return op->getParentOp()->getBlock(); })
					.Case([&rewriter](mlir::func::FuncOp op) {
						if (op.getBlocks().size() == 1) {
							auto insertion_point = rewriter.getInsertionPoint();
							auto *return_block = rewriter.createBlock(&op.getRegion());
							auto value = rewriter.create<mlir::py::ConstantOp>(
								op.getLoc(), rewriter.getNoneType());
							rewriter.create<mlir::func::ReturnOp>(
								op.getLoc(), mlir::ValueRange{ value });
							rewriter.setInsertionPoint(
								insertion_point->getBlock(), insertion_point);
							return return_block;
						}
						return &op.back();
					})
					.Default([](mlir::Operation *op) {
						TODO();
						return nullptr;
					});
			}

			mlir::LogicalResult matchAndRewrite(mlir::py::RaiseOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				if (auto exception = op.getException()) {
					rewriter.replaceOpWithNewOp<mlir::emitpybytecode::RaiseVarargs>(
						op, exception, op.getCause(), get_handler(op, rewriter));
				} else {
					rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ReRaiseOp>(
						op, get_handler(op, rewriter));
				}

				return success();
			}
		};

		struct ImportOpLowering : public mlir::OpRewritePattern<mlir::py::ImportOp>
		{
			using OpRewritePattern<mlir::py::ImportOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::ImportOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto name = op.getName();
				auto level = rewriter.create<mlir::emitpybytecode::ConstantOp>(op.getLoc(),
					op.getModule().getType(),
					rewriter.getUI32IntegerAttr(op.getLevel()));
				std::vector<mlir::Value> els;
				for (auto from : op.getFromList().getValues<mlir::StringRef>()) {
					els.push_back(rewriter.create<mlir::emitpybytecode::ConstantOp>(
						op.getLoc(), op.getModule().getType(), rewriter.getStringAttr(from)));
				}
				auto from_list = rewriter.create<mlir::emitpybytecode::BuildTuple>(
					op.getLoc(), op.getModule().getType(), els);
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ImportName>(
					op, op.getModule().getType(), name, level, from_list);

				return success();
			}
		};

		struct ImportAllOpLowering : public mlir::OpRewritePattern<mlir::py::ImportAllOp>
		{
			using OpRewritePattern<mlir::py::ImportAllOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::ImportAllOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ImportAll>(op, op.getModule());

				return success();
			}
		};

		struct ImportFromOpLowering : public mlir::OpRewritePattern<mlir::py::ImportFromOp>
		{
			using OpRewritePattern<mlir::py::ImportFromOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::ImportFromOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::ImportFrom>(
					op, op.getModule().getType(), op.getModule(), op.getName());

				return success();
			}
		};

		struct CastToBoolOpLowering : public mlir::OpRewritePattern<mlir::py::CastToBoolOp>
		{
			using OpRewritePattern<mlir::py::CastToBoolOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::CastToBoolOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::CastToBool>(
					op, op.getValue().getType(), op.getValue());
				return success();
			}
		};

		struct YieldOpLowering : public mlir::OpRewritePattern<mlir::py::YieldOp>
		{
			using OpRewritePattern<mlir::py::YieldOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::YieldOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::Yield>(
					op, op.getValue().getType(), op.getValue());
				return success();
			}
		};

		struct YieldFromOpLowering : public mlir::OpRewritePattern<mlir::py::YieldFromOp>
		{
			using OpRewritePattern<mlir::py::YieldFromOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::YieldFromOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				auto iterator = rewriter.create<mlir::emitpybytecode::YieldFromIter>(
					op.getLoc(), op.getIterable().getType(), op.getIterable());
				auto value =
					rewriter.create<mlir::py::ConstantOp>(op.getLoc(), rewriter.getNoneType());

				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::YieldFrom>(
					op, iterator.getType(), iterator, value);

				return success();
			}
		};

		struct UnpackSequenceOpLowering : public mlir::OpRewritePattern<mlir::py::UnpackSequenceOp>
		{
			using OpRewritePattern<mlir::py::UnpackSequenceOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::UnpackSequenceOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::UnpackSequenceOp>(
					op, op.getUnpackedValues().getType(), op.getIterable());
				return success();
			}
		};

		struct GetAwaitableOpLowering : public mlir::OpRewritePattern<mlir::py::GetAwaitableOp>
		{
			using OpRewritePattern<mlir::py::GetAwaitableOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(mlir::py::GetAwaitableOp op,
				mlir::PatternRewriter &rewriter) const final
			{
				rewriter.replaceOpWithNewOp<mlir::emitpybytecode::GetAwaitableOp>(
					op, op.getIterator().getType(), op.getIterable());
				return success();
			}
		};

		struct PythonToPythonBytecodePass
			: public PassWrapper<PythonToPythonBytecodePass, OperationPass<ModuleOp>>
		{
			MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PythonToPythonBytecodePass)

			void getDependentDialects(DialectRegistry &registry) const override
			{
				registry.insert<PythonDialect, emitpybytecode::EmitPythonBytecodeDialect>();
			}

			StringRef getArgument() const final { return "python-to-pythonbytecode"; }

			void runOnOperation() final;
		};
	}// namespace

	void PythonToPythonBytecodePass::runOnOperation()
	{
		ConversionTarget target(getContext());
		target.addLegalDialect<emitpybytecode::EmitPythonBytecodeDialect, mlir::BuiltinDialect>();

		target.addLegalOp<mlir::cf::BranchOp>();
		target.addLegalOp<mlir::func::ReturnOp>();
		target.addDynamicallyLegalOp<mlir::func::FuncOp>([](mlir::func::FuncOp op) {
			// don't convert this special function, which is the entry point of a module
			return op.isPrivate() && op.getSymName() == "__hidden_init__";
		});
		target.addIllegalDialect<PythonDialect>();

		mlir::RewritePatternSet patterns(&getContext());
		patterns.add<ConstantLoadLowering,
			LoadFastLowering,
			LoadNameLowering,
			LoadGlobalLowering,
			LoadDerefLowering,
			UnpackSequenceOpLowering>(&getContext());
		patterns.add<StoreFastLowering, StoreGlobalLowering, StoreNameLowering, StoreDerefLowering>(
			&getContext());
		patterns.add<DeleteFastLowering, DeleteGlobalLowering, DeleteNameLowering>(&getContext());
		patterns.add<CallFunctionLowering, FuncOpLowering, MakeFunctionOpLowering>(&getContext());
		patterns.add<BinaryAddOpLowering,
			BinarySubtractOpLowering,
			BinaryModuloOpLowering,
			BinaryMultiplyOpLowering,
			BinaryExpOpLowering,
			BinaryDivOpLowering,
			BinaryFloorDivOpLowering,
			BinaryMatMulOpLowering,
			LeftShiftOpLowering,
			RightShiftOpLowering,
			LogicalAndOpLowering,
			LogicalOrOpLowering,
			LogicalXorOpLowering>(&getContext());
		patterns.add<InplaceOpLowering>(&getContext());
		patterns
			.add<ConditionalBranchOpLowering, CondBranchSubclassOpLowering, CastToBoolOpLowering>(
				&getContext());
		patterns.add<CompareOpLowering>(&getContext());
		patterns.add<LoadAssertionErrorOpLowering, RaiseOpLowering>(&getContext());
		patterns.add<PositiveOpLowering, NegativeOpLowering, InvertOpLowering, NotOpLowering>(
			&getContext());
		patterns.add<BuildDictOpLowering,
			DictAddOpLowering,
			BuildListOpLowering,
			ListAppendOpLowering,
			BuildTupleOpLowering,
			BuildSetOpLowering,
			SetAddOpLowering,
			BuildStringOpLowering,
			FormatValueOpLowering>(&getContext());
		patterns.add<LoadAttributeOpLowering, LoadMethodOpLowering>(&getContext());
		patterns
			.add<BinarySubscriptOpLowering, StoreSubscriptOpLowering, DeleteSubscriptOpLowering>(
				&getContext());
		patterns.add<StoreAttributeOpLowering, BuildSliceOpLowering>(&getContext());
		patterns.add<ForLoopOpLowering, WhileOpLowering>(&getContext());
		patterns.add<TryOpLowering,
			WithOpLowering,
			WithExceptStartOpLowering,
			ClearExceptionStateOpLowering>(&getContext());
		patterns.add<ImportOpLowering, ImportFromOpLowering, ImportAllOpLowering>(&getContext());
		patterns.add<ClassDefinitionOpLowering>(&getContext());
		patterns.add<YieldOpLowering, YieldFromOpLowering>(&getContext());
		patterns.add<GetAwaitableOpLowering>(&getContext());

		if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
			signalPassFailure();
		}

		mlir::IRRewriter rewriter{ &getContext() };
		(void)eraseUnreachableBlocks(rewriter, getOperation()->getRegions());
	}

	std::unique_ptr<Pass> createPythonToPythonBytecodePass()
	{
		return std::make_unique<PythonToPythonBytecodePass>();
	}

}// namespace py
}// namespace mlir