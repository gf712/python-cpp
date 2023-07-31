include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Python_Dialect : Dialect {
    let name = "python";

    let summary = "python-cpp MLIR high level dialect";

    let description = "TODO";

    let cppNamespace = "::mlir::py";

    let useDefaultTypePrinterParser = 1;
}

def Python_CmpPredicateAttr : I64EnumAttr<
    "CmpPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>,
      I64EnumAttrCase<"lt", 2>,
      I64EnumAttrCase<"le", 3>,
      I64EnumAttrCase<"gt", 4>,
      I64EnumAttrCase<"ge", 5>,
      I64EnumAttrCase<"is", 6>,
      I64EnumAttrCase<"isnot", 7>,
      I64EnumAttrCase<"in", 8>,
      I64EnumAttrCase<"notin", 9>,
    ]> {
  let cppNamespace = "::mlir::py";
}

def Python_LoopOpKindAttr : I64EnumAttr<
    "LoopOpKind", "",
    [
      I64EnumAttrCase<"continue_", 0>,
      I64EnumAttrCase<"break_", 1>,
    ]> {
  let cppNamespace = "::mlir::py";
}