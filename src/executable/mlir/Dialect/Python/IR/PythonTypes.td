include "Python/IR/PythonBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"


class Python_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Python_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Python_PyObjectType: Python_Type<"PyObject", "object", []> {
    let summary = "Python object type";

    let description = "Represents the Python object type";
}