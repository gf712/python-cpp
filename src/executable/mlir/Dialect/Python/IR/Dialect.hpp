#pragma once

#include "mlir/IR/Dialect.h"
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

}// namespace mlir::py