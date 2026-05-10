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
	namespace {
		// MLIR 23 removed RegionBranchPoint::getRegionOrNull(). The new API exposes
		// the terminator op via getTerminatorPredecessorOrNull(); the region is the
		// terminator's parent region. Returns nullptr when the branch point is the
		// parent op.
		mlir::Region *predecessor_region(mlir::RegionBranchPoint point)
		{
			if (point.isParent()) { return nullptr; }
			auto term = point.getTerminatorPredecessorOrNull();
			if (!term) { return nullptr; }
			return term.getOperation()->getParentRegion();
		}
	}// namespace

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

	mlir::LogicalResult ConstantOp::verify()
	{
		mlir::Attribute attr = getValue();
		// Accepted attribute kinds correspond to the constant kinds the
		// ConstantOp builders can construct, plus EllipsisAttr (lowered to
		// LoadEllipsisOp by the conversion pass).
		if (mlir::isa<mlir::FloatAttr,
				mlir::BoolAttr,
				mlir::UnitAttr,
				mlir::StringAttr,
				mlir::IntegerAttr,
				mlir::DenseIntElementsAttr,
				mlir::ArrayAttr,
				EllipsisAttr>(attr)) {
			return mlir::success();
		}
		return emitOpError() << "value attribute has unsupported kind: " << attr;
	}

	SuccessorOperands CondBranchSubclassOp::getSuccessorOperands(unsigned index)
	{
		assert(index < getNumSuccessors() && "invalid successor index");
		return SuccessorOperands(
			index == 0 ? getTrueDestOperandsMutable() : getFalseDestOperandsMutable());
	}

	namespace {
		// Static getSuccessorInputs implementation shared by all four
		// RegionBranchOpInterface ops in this dialect (WhileOp, ForLoopOp,
		// TryOp, TryHandlerScope). MLIR 23 split the (region, block-args) pair
		// that used to be carried by RegionSuccessor: the region is now in
		// RegionSuccessor and the inputs come from this method. None of these
		// ops thread operands through the parent-branch successor (they all
		// return PyObject results that are produced by the regions, not
		// forwarded by the op itself), so the parent case returns empty.
		mlir::ValueRange region_or_block_arguments(mlir::Operation *,
			mlir::RegionSuccessor successor)
		{
			if (successor.isParent()) { return mlir::ValueRange{}; }
			return successor.getSuccessor()->getArguments();
		}
	}// namespace

	// Based on CIR loop interface implementation
	void WhileOp::getSuccessorRegions(mlir::RegionBranchPoint point,
		llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions)
	{
		// Branching to first region: go to condition.
		if (point.isParent()) {
			regions.emplace_back(&getCondition());
		}
		// Branching from condition: go to body or, exit or orelse if non-empty.
		else if (predecessor_region(point) == &getCondition()) {
			if (getOrelse().empty()) {
				regions.emplace_back(RegionSuccessor::parent());
			} else {
				regions.emplace_back(&getOrelse());
			}
			regions.emplace_back(&getBody());
		}
		// Branching from body: go to condition.
		else if (predecessor_region(point) == &getBody()) {
			regions.emplace_back(&getCondition());
		}
		// Branching from orelse - can't go anywhere else.
		else if (predecessor_region(point) == &getOrelse()) {
		} else {
			llvm_unreachable("unexpected branch origin");
		}
	}

	void ForLoopOp::getSuccessorRegions(mlir::RegionBranchPoint point,
		llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions)
	{
		// Branching to first region: go to step.
		if (point.isParent()) {
			regions.emplace_back(&getStep());
		}
		// Branching from condition: go to body or, exit or orelse if non-empty.
		else if (predecessor_region(point) == &getStep()) {
			if (getOrelse().empty()) {
				regions.emplace_back(RegionSuccessor::parent());
			} else {
				regions.emplace_back(&getOrelse());
			}
			regions.emplace_back(&getBody());
		}
		// Branching from body: go to step.
		else if (predecessor_region(point) == &getBody()) {
			regions.emplace_back(&getStep());
		}
		// Branching from orelse - can't go anywhere else.
		else if (predecessor_region(point) == &getOrelse()) {
		} else {
			llvm_unreachable("unexpected branch origin");
		}
	}

	mlir::ValueRange WhileOp::getSuccessorInputs(mlir::RegionSuccessor successor)
	{
		return region_or_block_arguments(getOperation(), successor);
	}

	mlir::ValueRange ForLoopOp::getSuccessorInputs(mlir::RegionSuccessor successor)
	{
		return region_or_block_arguments(getOperation(), successor);
	}

	void TryOp::getSuccessorRegions(mlir::RegionBranchPoint point,
		llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions)
	{
		// Branching to first region: go to try body.
		if (point.isParent()) {
			regions.emplace_back(&getBody());
		}
		// Branching from try body: go to first handler and orelse block if non-empty, if there no
		// handlers go to finally
		else if (predecessor_region(point) == &getBody()) {
			if (!getHandlers().empty()) {
				regions.emplace_back(&getHandlers().front());
				if (!getOrelse().empty()) { regions.emplace_back(&getOrelse()); }
			} else {
				assert(getOrelse().empty());
				regions.emplace_back(&getFinally());
			}
		}
		// Branching from handler: go to next handler if there is one, if not go to finally.
		else if (auto it = std::find_if(getHandlers().begin(),
					 getHandlers().end(),
					 [&point](
						 mlir::Region &handler) { return predecessor_region(point) == &handler; });
			it != getHandlers().end()) {
			if (std::next(it) != getHandlers().end()) {
				it++;
				regions.emplace_back(&*it);
			}
			if (!getFinally().empty()) { regions.emplace_back(&getFinally()); }
			// regions.emplace_back(getOperation()->getParentRegion());
		}
		// Branch from orelse: go to finally or parent
		else if (predecessor_region(point) == &getOrelse()) {
			if (!getFinally().empty()) {
				regions.emplace_back(&getFinally());
			} else {
				regions.emplace_back(getOperation()->getParentRegion());
			}
		}
		// Branch from finally: go to parent
		else if (predecessor_region(point) == &getFinally()) {
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

	mlir::ValueRange TryOp::getSuccessorInputs(mlir::RegionSuccessor successor)
	{
		return region_or_block_arguments(getOperation(), successor);
	}

	mlir::ValueRange TryHandlerScope::getSuccessorInputs(mlir::RegionSuccessor successor)
	{
		return region_or_block_arguments(getOperation(), successor);
	}

	void TryHandlerScope::getSuccessorRegions(mlir::RegionBranchPoint point,
		llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions)
	{
		if (predecessor_region(point) == &getCond()) { regions.emplace_back(&getHandler()); }
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
