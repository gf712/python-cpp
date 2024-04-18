#ifndef PYTHON_BASE
#define PYTHON_BASE

include "mlir/IR/OpBase.td"

def Python_Dialect : Dialect {
    let name = "python";

    let summary = "python-cpp high level dialect";

    let description = [{
This dialect is almost a direct representation of the AST.
However, it adds additional information, such as the variable
scope (local, global or nested), and some lower level concepts
that map more directly to the python-cpp Bytecode VM.
    }];

    let cppNamespace = "::mlir::py";

    let useDefaultTypePrinterParser = 1;

    let useDefaultAttributePrinterParser = 1;
}

#endif // PYTHON_BASE