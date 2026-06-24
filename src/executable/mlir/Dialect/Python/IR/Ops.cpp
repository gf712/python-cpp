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
		mlir::py::EllipsisAttr value)
	{
		ConstantOp::build(builder, state, PyObjectType::get(builder.getContext()), value);
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

	mlir::LogicalResult FunctionCallOp::verify()
	{
		// Two distinct shapes share this op:
		//
		// 1. Plain kwargs: keywords[i] names kwargs[i], so the two lists
		//    are parallel and must agree in length.
		//
		// 2. Expansion: when requires_{args,kwargs}_expansion is set,
		//    the corresponding operand list holds at most one value (a
		//    tuple or dict to splat), and keywords is empty regardless
		//    of how many actual keyword names the call will produce at
		//    runtime. MLIRGenerator currently still emits a kwargs
		//    operand alongside an args-expansion-only call, so we can't
		//    use the parallel rule in that case.
		if (getRequiresArgsExpansion()) {
			if (getArgs().size() > 1) {
				return emitOpError()
					   << "requires_args_expansion expects at most one args operand, got "
					   << getArgs().size();
			}
		}
		if (getRequiresKwargsExpansion()) {
			if (getKwargs().size() > 1) {
				return emitOpError()
					   << "requires_kwargs_expansion expects at most one kwargs operand, got "
					   << getKwargs().size();
			}
		}
		if (getRequiresArgsExpansion() || getRequiresKwargsExpansion()) { return mlir::success(); }
		const auto keywords_size = getKeywords().size();
		const auto kwargs_size = getKwargs().size();
		if (keywords_size != kwargs_size) {
			return emitOpError() << "has " << keywords_size << " keyword name(s) but "
								 << kwargs_size << " kwargs value(s)";
		}
		return mlir::success();
	}

	mlir::LogicalResult UnpackSequenceOp::verify()
	{
		// Variadic results, but a zero-result unpack is meaningless: there
		// would be no Python-level target receiving the unpacked values
		// and the bytecode emitter would still pay for the iterator
		// machinery. MLIRGenerator only emits this for sequence-assignment
		// targets, which always carry at least one binding.
		if (getUnpackedValues().empty()) {
			return emitOpError() << "must produce at least one unpacked value";
		}
		return mlir::success();
	}

	mlir::LogicalResult ClassDefinitionOp::verify()
	{
		// keywords[i] names kwargs[i] — the two lists must agree in length.
		// Unlike py.call, ClassDefinitionOp has no expansion flag, so the
		// parallel rule always applies.
		const auto keywords_size = getKeywords().size();
		const auto kwargs_size = getKwargs().size();
		if (keywords_size != kwargs_size) {
			return emitOpError() << "has " << keywords_size << " keyword name(s) but "
								 << kwargs_size << " kwargs value(s)";
		}
		return mlir::success();
	}

	mlir::LogicalResult BuildDictOp::verify()
	{
		// SameVariadicOperandSize already enforces keys.size() == values.size().
		// requires_expansion is one bool per kv pair.
		const auto expansion_size = getRequiresExpansion().size();
		const auto keys_size = getKeys().size();
		if (expansion_size != keys_size) {
			return emitOpError() << "requires_expansion has " << expansion_size
								 << " entries but op has " << keys_size << " key/value pairs";
		}
		return mlir::success();
	}

	namespace {
		template<typename Op> mlir::LogicalResult verify_elementwise_expansion(Op op)
		{
			const auto expansion_size = op.getRequiresExpansion().size();
			const auto elements_size = op.getElements().size();
			if (expansion_size != elements_size) {
				return op.emitOpError() << "requires_expansion has " << expansion_size
										<< " entries but op has " << elements_size << " elements";
			}
			return mlir::success();
		}
	}// namespace

	mlir::LogicalResult BuildListOp::verify() { return verify_elementwise_expansion(*this); }
	mlir::LogicalResult BuildTupleOp::verify() { return verify_elementwise_expansion(*this); }
	mlir::LogicalResult BuildSetOp::verify() { return verify_elementwise_expansion(*this); }

	SuccessorOperands CondBranchSubclassOp::getSuccessorOperands(unsigned index)
	{
		assert(index < getNumSuccessors() && "invalid successor index");
		return SuccessorOperands(
			index == 0 ? getTrueDestOperandsMutable() : getFalseDestOperandsMutable());
	}

	namespace {
		// Shared getSuccessorInputs implementation for all four
		// RegionBranchOpInterface ops in this dialect (WhileOp, ForLoopOp,
		// TryOp, TryHandlerOp). MLIR 23 split the (region, block-args)
		// pair that used to be carried by RegionSuccessor: the region is now
		// in RegionSuccessor and the inputs come from this method. The
		// inputs MLIR expects here are the operands the parent op forwards
		// INTO the destination (which the verifier matches against the op's
		// operand list along each control-flow edge).
		//
		// None of these ops actually forward operands into their regions or
		// out of them: the for-loop iterator is consumed by FOR_ITER inside
		// `step`; the while condition is computed inside `condition`; try's
		// regions communicate via the exception state, not block args; and
		// the ops' results are produced by the BranchYieldOp terminator,
		// not threaded through the parent successor. Return empty in all
		// cases.
		mlir::ValueRange region_or_block_arguments(mlir::Operation *, mlir::RegionSuccessor)
		{
			return mlir::ValueRange{};
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
				regions.emplace_back(RegionSuccessor(getOperation()));
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
				regions.emplace_back(RegionSuccessor(getOperation()));
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

	void BranchYieldOp::getSuccessorRegions(llvm::ArrayRef<mlir::Attribute> operands,
		llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions)
	{
		static_assert(BranchYieldOp::hasTrait<
			mlir::OpTrait::HasParent<TryOp, ForLoopOp, WithOp, WhileOp, TryHandlerOp>::Impl>());

		if (getKind().has_value()) { return; }

		auto result = llvm::TypeSwitch<Operation *, LogicalResult>(getOperation()->getParentOp())
						  .Case<TryOp>([this, &regions](TryOp op) -> LogicalResult {
							  // Fallthrough (no exception) successors for a yield inside
							  // a TryOp. The exception → handler edges are not modeled
							  // here; they're induced by the exception state, not by a
							  // BranchYieldOp. Successor of yield from:
							  //   body    -> orelse if non-empty, else finally if
							  //              non-empty, else parent.
							  //   handler -> finally if non-empty, else parent.
							  //   orelse  -> finally if non-empty, else parent.
							  //   finally -> parent.
							  // Matches the "exit-to-parent" convention used by
							  // TryOp::getSuccessorRegions (emplaces the containing
							  // region rather than RegionSuccessor::parent()).
							  auto *parent_region = getOperation()->getParentRegion();
							  auto exit_to_finally_or_parent = [&] {
								  if (!op.getFinally().empty()) {
									  regions.emplace_back(&op.getFinally());
								  } else {
									  regions.emplace_back(op->getParentRegion());
								  }
							  };
							  if (parent_region == &op.getBody()) {
								  if (!op.getOrelse().empty()) {
									  regions.emplace_back(&op.getOrelse());
								  } else {
									  exit_to_finally_or_parent();
								  }
							  } else if (parent_region == &op.getOrelse()) {
								  exit_to_finally_or_parent();
							  } else if (parent_region == &op.getFinally()) {
								  regions.emplace_back(op->getParentRegion());
							  } else {
								  bool in_handler = false;
								  for (mlir::Region &handler : op.getHandlers()) {
									  if (parent_region == &handler) {
										  in_handler = true;
										  break;
									  }
								  }
								  if (!in_handler) { llvm_unreachable("unexpected branch origin"); }
								  exit_to_finally_or_parent();
							  }
							  return success();
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
						  .Case<TryHandlerOp>([this, &regions](TryHandlerOp op) -> LogicalResult {
							  // TryHandlerOp models a single except-clause: cond is
							  // the type-match test, handler is the body run on match.
							  // Yield from cond can fall through to the handler (match)
							  // or out to the parent (no match - try the next clause or
							  // re-raise). Yield from handler exits the scope.
							  auto *parent_region = getOperation()->getParentRegion();
							  if (parent_region == &op.getCond()) {
								  regions.emplace_back(&op.getHandler());
								  regions.emplace_back(op->getParentRegion());
							  } else if (parent_region == &op.getHandler()) {
								  regions.emplace_back(op->getParentRegion());
							  } else {
								  llvm_unreachable("unexpected branch origin");
							  }
							  return success();
						  })
						  .Default([](Operation *) -> LogicalResult {
							  // Unreachable in verified IR: the static_assert above
							  // constrains BranchYieldOp's parent op to one of the
							  // five Cases handled. A failure here means an
							  // unverified or malformed op slipped past verification.
							  llvm_unreachable("BranchYieldOp has unexpected parent op kind");
						  });

		assert(result.succeeded());
	}

	mlir::ValueRange TryOp::getSuccessorInputs(mlir::RegionSuccessor successor)
	{
		return region_or_block_arguments(getOperation(), successor);
	}

	mlir::ValueRange TryHandlerOp::getSuccessorInputs(mlir::RegionSuccessor successor)
	{
		return region_or_block_arguments(getOperation(), successor);
	}

	void TryHandlerOp::getSuccessorRegions(mlir::RegionBranchPoint point,
		llvm::SmallVectorImpl<mlir::RegionSuccessor> &regions)
	{
		if (predecessor_region(point) == &getCond()) { regions.emplace_back(&getHandler()); }
		regions.emplace_back(getOperation()->getParentRegion());
	}

	namespace {
		// Forward the value of a preceding py.store_fast of the same name in
		// the same block when no intervening op kills the binding. Locals
		// can only be touched by store_fast / delete_fast in the current
		// FuncOp, so most ops (including FunctionCall) are safe to walk
		// past - with one exception: ops with regions (py.for_loop /
		// py.while / py.try / py.with / py.if-like ops) may contain a
		// store_fast or delete_fast targeting this name in their bodies.
		// We don't recurse into those regions yet, so conservatively bail
		// if any region-bearing op sits between the candidate store and the
		// load.
		struct ForwardStoreFastToLoadFast : public mlir::OpRewritePattern<LoadFastOp>
		{
			using mlir::OpRewritePattern<LoadFastOp>::OpRewritePattern;

			mlir::LogicalResult matchAndRewrite(LoadFastOp load,
				mlir::PatternRewriter &rewriter) const final
			{
				const auto name = load.getName();
				for (mlir::Operation *prev = load->getPrevNode(); prev != nullptr;
					prev = prev->getPrevNode()) {
					if (auto store = mlir::dyn_cast<StoreFastOp>(prev)) {
						if (store.getName() == name) {
							rewriter.replaceOp(load, store.getValue());
							return mlir::success();
						}
					}
					if (auto del = mlir::dyn_cast<DeleteFastOp>(prev)) {
						if (del.getName() == name) {
							// The binding was deleted between the store
							// and the load - can't forward.
							return mlir::failure();
						}
					}
					if (prev->getNumRegions() > 0) {
						// A nested region (loop body, try/with body, ...)
						// might contain a store or delete of `name`. Be
						// conservative and bail rather than try to prove it
						// doesn't.
						return mlir::failure();
					}
				}
				return mlir::failure();
			}
		};
	}// namespace

	void LoadFastOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
		mlir::MLIRContext *context)
	{
		patterns.add<ForwardStoreFastToLoadFast>(context);
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
