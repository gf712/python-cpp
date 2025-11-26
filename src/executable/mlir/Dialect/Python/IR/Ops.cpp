#include "Dialect.hpp"
#include "PythonAttributes.hpp"
#include "PythonOps.hpp"
#include "PythonTypes.hpp"

#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#include "Python/IR/Dialect.cpp.inc"

namespace mlir {
namespace py {
	void PythonDialect::initialize()
	{
		addOperations<
#define GET_OP_LIST
#include "Python/IR/Ops.cpp.inc"
			>();

		addTypes<
#define GET_TYPEDEF_LIST
#include "Python/IR/PythonTypes.cpp.inc"
			>();

		addAttributes<
#define GET_ATTRDEF_LIST
#include "Python/IR/PythonAttributes.cpp.inc"
			>();
	}

	void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value)
	{
		ConstantOp::build(builder,
			state,
			PyObjectType::get(builder.getContext()),
			FloatAttr::get(builder.getF64Type(), value));
	}

	void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, bool value)
	{
		ConstantOp::build(
			builder, state, PyObjectType::get(builder.getContext()), builder.getBoolAttr(value));
	}

	void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::NoneType)
	{
		ConstantOp::build(
			builder, state, PyObjectType::get(builder.getContext()), builder.getUnitAttr());
	}

	void ConstantOp::build(mlir::OpBuilder &builder,
		mlir::OperationState &state,
		mlir::StringAttr str)
	{
		ConstantOp::build(builder, state, PyObjectType::get(builder.getContext()), str);
	}

	void ConstantOp::build(mlir::OpBuilder &builder,
		mlir::OperationState &state,
		mlir::IntegerAttr int_attr)
	{
		ConstantOp::build(builder, state, PyObjectType::get(builder.getContext()), int_attr);
	}

	void ConstantOp::build(mlir::OpBuilder &builder,
		mlir::OperationState &state,
		std::vector<std::byte> bytes)
	{
		auto byte_array_attr = mlir::DenseIntElementsAttr::get(
			mlir::VectorType::get(
				static_cast<int64_t>(bytes.size()), builder.getIntegerType(8, false)),
			mlir::ArrayRef<unsigned char>{
				llvm::bit_cast<unsigned char *>(bytes.data()), bytes.size() });
		ConstantOp::build(builder, state, PyObjectType::get(builder.getContext()), byte_array_attr);
	}

	void ConstantOp::build(mlir::OpBuilder &builder,
		mlir::OperationState &state,
		mlir::py::PyEllipsisType)
	{
		ConstantOp::build(builder,
			state,
			PyObjectType::get(builder.getContext()),
			EllipsisAttr::get(builder.getContext()));
	}

	void ConstantOp::build(mlir::OpBuilder &builder,
		mlir::OperationState &state,
		mlir::ArrayRef<mlir::Attribute> elements)
	{
		ConstantOp::build(builder,
			state,
			PyObjectType::get(builder.getContext()),
			mlir::ArrayAttr::get(builder.getContext(), std::move(elements)));
	}

	EllipsisAttr EllipsisAttr::get(mlir::MLIRContext *context)
	{
		return mlir::detail::AttributeUniquer::get<mlir::py::EllipsisAttr>(context);
	}

	SuccessorOperands CondBranchSubclassOp::getSuccessorOperands(unsigned index)
	{
		assert(index < getNumSuccessors() && "invalid successor index");
		return SuccessorOperands(
			index == 0 ? getTrueDestOperandsMutable() : getFalseDestOperandsMutable());
	}

	// Based on CIR loop interface implementation
	void WhileOp::getSuccessorRegions(mlir::RegionBranchPoint point,
		llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions)
	{
		// Branching to first region: go to condition.
		if (point.isParent()) {
			regions.emplace_back(&getCondition(), getCondition().getArguments());
		}
		// Branching from condition: go to body or, exit or orelse if non-empty.
		else if (point.getRegionOrNull() == &getCondition()) {
			if (getOrelse().empty()) {
				regions.emplace_back(RegionSuccessor(getOperation()->getResults()));
			} else {
				regions.emplace_back(&getOrelse(), getOrelse().getArguments());
			}
			regions.emplace_back(&getBody(), getBody().getArguments());
		}
		// Branching from body: go to condition.
		else if (point.getRegionOrNull() == &getBody()) {
			regions.emplace_back(&getCondition(), getCondition().getArguments());
		}
		// Branching from orelse - can't go anywhere else.
		else if (point.getRegionOrNull() == &getOrelse()) {
		} else {
			llvm_unreachable("unexpected branch origin");
		}
	}

	void ForLoopOp::getSuccessorRegions(mlir::RegionBranchPoint point,
		llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions)
	{
		// Branching to first region: go to step.
		if (point.isParent()) {
			regions.emplace_back(&getStep(), getStep().getArguments());
		}
		// Branching from condition: go to body or, exit or orelse if non-empty.
		else if (point.getRegionOrNull() == &getStep()) {
			if (getOrelse().empty()) {
				regions.emplace_back(getOperation()->getResults());
			} else {
				regions.emplace_back(&getOrelse(), getOrelse().getArguments());
			}
			regions.emplace_back(&getBody(), getBody().getArguments());
		}
		// Branching from body: go to step.
		else if (point.getRegionOrNull() == &getBody()) {
			regions.emplace_back(&getStep(), getStep().getArguments());
		}
		// Branching from orelse - can't go anywhere else.
		else if (point.getRegionOrNull() == &getOrelse()) {
		} else {
			llvm_unreachable("unexpected branch origin");
		}
	}

	void TryOp::getSuccessorRegions(mlir::RegionBranchPoint point,
		llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions)
	{
		// Branching to first region: go to try body.
		if (point.isParent()) {
			regions.emplace_back(&getBody(), getBody().getArguments());
		}
		// Branching from try body: go to first handler and orelse block if non-empty, if there no
		// handlers go to finally
		else if (point.getRegionOrNull() == &getBody()) {
			if (!getHandlers().empty()) {
				regions.emplace_back(&getHandlers().front(), getHandlers().front().getArguments());
				if (!getOrelse().empty()) {
					regions.emplace_back(&getOrelse(), getOrelse().getArguments());
				}
			} else {
				assert(getOrelse().empty());
				regions.emplace_back(&getFinally(), getFinally().getArguments());
			}
		}
		// Branching from handler: go to next handler if there is one, if not go to finally.
		else if (auto it = std::find_if(getHandlers().begin(),
					 getHandlers().end(),
					 [&point](
						 mlir::Region &handler) { return point.getRegionOrNull() == &handler; });
				 it != getHandlers().end()) {
			if (std::next(it) != getHandlers().end()) {
				it++;
				regions.emplace_back(&*it, it->getArguments());
			}
			if (!getFinally().empty()) {
				regions.emplace_back(&getFinally(), getFinally().getArguments());
			}
			// regions.emplace_back(getOperation()->getParentRegion());
		}
		// Branch from orelse: go to finally or parent
		else if (point.getRegionOrNull() == &getOrelse()) {
			if (!getFinally().empty()) {
				regions.emplace_back(&getFinally(), getFinally().getArguments());
			} else {
				regions.emplace_back(getOperation()->getParentRegion());
			}
		}
		// Branch from finally: go to parent
		else if (point.getRegionOrNull() == &getFinally()) {
			regions.emplace_back(getOperation()->getParentRegion());
		}
	}

	void ControlFlowYield::getSuccessorRegions(llvm::ArrayRef<mlir::Attribute> operands,
		llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions)
	{
		static_assert(ControlFlowYield::hasTrait<
			mlir::OpTrait::HasParent<TryOp, ForLoopOp, WithOp, WhileOp, TryHandlerScope>::Impl>());

		if (getKind().has_value()) { return; }

		auto result =
			llvm::TypeSwitch<Operation *, LogicalResult>(getOperation()->getParentOp())
				.Case<TryOp>([&regions](TryOp op) -> LogicalResult {
					// regions.emplace_back(&op.getRegion());
					llvm_unreachable("TODO");
					return failure();
				})
				.Case<ForLoopOp>([this, &regions](ForLoopOp op) -> LogicalResult {
					if (getOperation()->getParentRegion() == &op.getStep()) {
						regions.emplace_back(&op.getBody());
					} else if (getOperation()->getParentRegion() == &op.getBody()) {
						regions.emplace_back(&op.getStep());
					} else if (getOperation()->getParentRegion() == &op.getOrelse()) {
					} else {
						llvm_unreachable("unexpected branch origin");
					}
					return success();
				})
				.Case<WithOp>([&regions](WithOp op) -> LogicalResult {
					regions.emplace_back(op->getParentRegion());
					return success();
				})
				.Case<WhileOp>([this, &regions](WhileOp op) -> LogicalResult {
					if (getOperation()->getParentRegion() == &op.getCondition()) {
						regions.emplace_back(&op.getBody());
					} else if (getOperation()->getParentRegion() == &op.getBody()) {
						regions.emplace_back(&op.getCondition());
					} else if (getOperation()->getParentRegion() == &op.getOrelse()) {
					} else {
						llvm_unreachable("unexpected branch origin");
					}
					return success();
				})
				.Case<TryHandlerScope>([this, &regions](TryHandlerScope op) -> LogicalResult {
					llvm_unreachable("todo");
					return failure();
				})
				.Default([](Operation *) -> LogicalResult {
					llvm_unreachable("TODO");
					std::abort();
					return failure();
				});

		assert(result.succeeded());
	}

	void TryHandlerScope::getSuccessorRegions(mlir::RegionBranchPoint point,
		llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions)
	{
		if (point.getRegionOrNull() == &getCond()) {
			regions.emplace_back(&getHandler(), getHandler().getArguments());
		}
		regions.emplace_back(getOperation()->getParentRegion());
	}
}// namespace py
}// namespace mlir

#define GET_OP_CLASSES
#include "Python/IR/Ops.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Python/IR/PythonTypes.cpp.inc"

#include "Python/IR/PythonOpsEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Python/IR/PythonAttributes.cpp.inc"
