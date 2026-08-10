#ifndef PYTHON_INTERFACES
#define PYTHON_INTERFACES

include "mlir/IR/OpBase.td"

// Shared by the ops that model a Python loop statement: py.for_loop and
// py.while.
//
// What the passes actually need from a loop is not "is it a loop" but its two
// control-flow-distinct regions, because Python treats them differently:
//
//   * the *body* is what `break`/`continue` inside it bind to. Its terminating
//     py.br_yield ops (kindless for fallthrough, kinded for break/continue) are
//     consumed by this loop's own lowering.
//
//   * the *orelse* is NOT part of the loop. It runs once, after a normal exit,
//     so a `break`/`continue` written there binds to the *enclosing* loop and
//     must be left for that loop's pattern. Only the orelse's kindless
//     completion yield belongs to this loop, as a branch to its exit.
//
// Before this interface existed both distinctions were spelled out as
// `isa<ForLoopOp>` / `isa<WhileOp>` pairs at each site, which is how the
// exemption in replace_loop_branch_yields came to cover try/with but not a
// nested loop's orelse — the two are indistinguishable in that form.
def PyLoopOpInterface : OpInterface<"PyLoopOpInterface"> {
    let description = [{
        A Python loop statement, whose body owns the break/continue written
        inside it and whose orelse belongs to the enclosing loop.
    }];
    let cppNamespace = "::mlir::py";

    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                The loop body. `break`/`continue` yields directly in this region
                bind to this loop.
            }],
            /*retTy=*/"::mlir::Region &",
            /*methodName=*/"getLoopBodyRegion"
        >,
        InterfaceMethod<
            /*desc=*/[{
                The `else` clause, empty when the loop has none. Loop-control
                yields here bind to the *enclosing* loop, not this one.
            }],
            /*retTy=*/"::mlir::Region &",
            /*methodName=*/"getLoopOrelseRegion"
        >,
    ];

    let extraClassDeclaration = [{
        // True when `region` is this loop's orelse. Distinguishing by region and
        // not just by parent op is what keeps a loop's own body yields out of
        // scope when a walk is looking for a nested loop's orelse.
        bool isLoopOrelse(::mlir::Region *region) {
            return region == &getLoopOrelseRegion();
        }
    }];
}

#endif // PYTHON_INTERFACES
