include "mlir/Pass/PassBase.td"


def ConvertPythonToPythonBytecode : Pass<"convert-python-to-pythonbytecode"> {
  let summary = "Convert recognized Python ops to PythonCpp bytecode";
  let constructor = "mlir::py::createPythonToPythonBytecodePass()";
}