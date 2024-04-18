include "mlir/IR/EnumAttr.td"
include "Python/IR/PythonBase.td"

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

def Python_FormatValueConversionAttr : I64EnumAttr<
    "FormatValueConversion", "",
    [
      I64EnumAttrCase<"none", 0>,
      I64EnumAttrCase<"repr", 1>,
      I64EnumAttrCase<"string", 2>,
      I64EnumAttrCase<"ascii", 3>,
    ]> {
  let cppNamespace = "::mlir::py";
}

def Python_InplaceOpKindAttr : I64EnumAttr<
    "InplaceOpKind", "",
    [
      I64EnumAttrCase<"add", 0>,
      I64EnumAttrCase<"sub", 1>,
      I64EnumAttrCase<"mod", 2>,
      I64EnumAttrCase<"mul", 3>,
      I64EnumAttrCase<"exp", 4>,
      I64EnumAttrCase<"div", 5>,
      I64EnumAttrCase<"fldiv", 6>,
      I64EnumAttrCase<"mmul", 7>,
      I64EnumAttrCase<"lshift", 8>,
      I64EnumAttrCase<"rshift", 9>,
      I64EnumAttrCase<"and_", 10>,
      I64EnumAttrCase<"or_", 11>,
      I64EnumAttrCase<"xor_", 12>,
    ]> {
  let cppNamespace = "::mlir::py";
}

class Python_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                  string baseCppClass = "::mlir::Attribute">
    : AttrDef<Python_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
  let attrName = "python." # attrMnemonic;
}

def Python_EllipsisAttr : Python_Attr<"Ellipsis", "ellipsis"> {
  let summary = "An Attribute representing the python ellipsis singleton";

  let extraClassDeclaration = [{
    static EllipsisAttr get(MLIRContext *context);
  }];
}