#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Python/IR/Dialect.h.inc"

namespace mlir::py {

// Resource representing the Python interpreter's active-exception state.
// Used as the write-resource for Load* ops: a load that targets a name not
// bound in the current scope raises NameError / UnboundLocalError, which
// is a visible side effect. Modelling that effect as a MemWrite on this
// resource prevents canonicalize/DCE from eliminating a load whose result
// happens to be unused.
struct PythonExceptionStateResource
	: public ::mlir::SideEffects::Resource::Base<PythonExceptionStateResource>
{
	::mlir::StringRef getName() const override { return "PythonExceptionState"; }
};

namespace OpTrait {
	// Trait asserting that an op carries a non-empty StringAttr "name". Used
	// by the {Load,Store,Delete}{Name,Fast,Global,Deref} family in
	// PythonOps.td. The check is defensive: code-gen always supplies a
	// non-empty name, but a malformed parse or a future bug shouldn't
	// silently produce an op with no binding target.
	template<typename ConcreteType>
	class NamedOp : public ::mlir::OpTrait::TraitBase<ConcreteType, NamedOp>
	{
	  public:
		static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op)
		{
			auto name = op->getAttrOfType<::mlir::StringAttr>("name");
			if (!name) { return op->emitOpError("requires a 'name' StringAttr"); }
			if (name.getValue().empty()) {
				return op->emitOpError("requires a non-empty 'name' attribute");
			}
			return ::mlir::success();
		}
	};
}// namespace OpTrait

}// namespace mlir::py