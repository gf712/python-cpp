include "EmitPythonBytecodeBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "Python/IR/PythonTypes.td"

// Base class for EmitPythonBytecode dialect ops.
class EmitPythonBytecode_Op<string mnemonic, list<Trait> traits = []>
    : Op<EmitPythonBytecode_Dialect, mnemonic, traits>;

def EmitPythonBytecode_ConstantOp: EmitPythonBytecode_Op<"LOAD_CONST"> {
    let arguments = (ins AnyAttr:$value);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadEllipsisOp: EmitPythonBytecode_Op<"LOAD_ELLIPSIS"> {
    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadNameOp: EmitPythonBytecode_Op<"LOAD_NAME"> {
    let arguments = (ins StrAttr:$name);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadFastOp: EmitPythonBytecode_Op<"LOAD_FAST"> {
    let arguments = (ins StrAttr:$name);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadGlobalOp: EmitPythonBytecode_Op<"LOAD_GLOBAL"> {
    let arguments = (ins StrAttr:$name);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadDerefOp: EmitPythonBytecode_Op<"LOAD_DEREF"> {
    let arguments = (ins StrAttr:$name);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadClosureOp: EmitPythonBytecode_Op<"LOAD_CLOSURE"> {
    let arguments = (ins StrAttr:$name);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_StoreNameOp: EmitPythonBytecode_Op<"STORE_NAME"> {
    let arguments = (ins StrAttr:$name, Python_PyObjectType:$object);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_StoreGlobalOp: EmitPythonBytecode_Op<"STORE_GLOBAL"> {
    let arguments = (ins StrAttr:$name, Python_PyObjectType:$object);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_StoreFastOp: EmitPythonBytecode_Op<"STORE_FAST"> {
    let arguments = (ins StrAttr:$name, Python_PyObjectType:$object);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_StoreDerefOp: EmitPythonBytecode_Op<"STORE_DEREF"> {
    let arguments = (ins StrAttr:$name, Python_PyObjectType:$object);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_DeleteNameOp: EmitPythonBytecode_Op<"DELETE_NAME"> {
    let arguments = (ins StrAttr:$name);
}

def EmitPythonBytecode_DeleteFastOp: EmitPythonBytecode_Op<"DELETE_FAST"> {
    let arguments = (ins StrAttr:$name);
}

def EmitPythonBytecode_DeleteGlobalOp: EmitPythonBytecode_Op<"DELETE_GLOBAL"> {
    let arguments = (ins StrAttr:$name);
}

def EmitPythonBytecode_DeleteDerefOp: EmitPythonBytecode_Op<"DELETE_DEREF"> {
    let arguments = (ins StrAttr:$name);
}

def EmitPythonBytecode_FunctionCallOp: EmitPythonBytecode_Op<"CALL"> {
    let arguments = (ins Python_PyObjectType:$callee, Variadic<Python_PyObjectType>:$args);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_FunctionCallExOp: EmitPythonBytecode_Op<"CALL_EXPAND", [AttrSizedOperandSegments]> {
    let arguments = (ins Python_PyObjectType:$callee,
                         Optional<Python_PyObjectType>:$args,
                         Optional<Python_PyObjectType>:$kwargs);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_FunctionCallWithKeywordsOp: EmitPythonBytecode_Op<"CALL_KW", [AttrSizedOperandSegments]> {
    let arguments = (ins Python_PyObjectType:$callee,
                         Variadic<Python_PyObjectType>:$args,
                         Builtin_DenseStringElementsAttr:$keywords,
                         Variadic<Python_PyObjectType>:$kwargs);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_BinaryOp: EmitPythonBytecode_Op<"BINARY_OP"> {
    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs, UI8Attr:$operation_type);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_InplaceOp: EmitPythonBytecode_Op<"INPLACE_OP"> {
    let arguments = (ins Python_PyObjectType:$dst, Python_PyObjectType:$src, UI8Attr:$operation_type);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_JumpIfFalse: EmitPythonBytecode_Op<"JUMP_IF_FALSE", [
    AttrSizedOperandSegments, Terminator,
    DeclareOpInterfaceMethods<BranchOpInterface>]> {
    let arguments = (ins Python_PyObjectType:$cond,
                         Variadic<Python_PyObjectType>:$trueDestOperands,
                         Variadic<Python_PyObjectType>:$falseDestOperands);

    let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

    let builders = [
        OpBuilder<(ins "Value":$condition, "Block *":$trueDest,
                   "ValueRange":$trueOperands, "Block *":$falseDest,
                   "ValueRange":$falseOperands), [{
            build($_builder, $_state, condition, trueOperands, falseOperands, trueDest, falseDest);
        }]>
    ];

    let assemblyFormat = [{
        $cond `,`
        $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
        $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
        attr-dict
    }];
}

def EmitPythonBytecode_JumpIfNotException: EmitPythonBytecode_Op<"JUMP_IF_NOT_EXC", [
    AttrSizedOperandSegments, Terminator,
    DeclareOpInterfaceMethods<BranchOpInterface>]> {
    let arguments = (ins Python_PyObjectType:$object_type,
                         Variadic<Python_PyObjectType>:$trueDestOperands,
                         Variadic<Python_PyObjectType>:$falseDestOperands);

    let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

    let assemblyFormat = [{
        $object_type `,`
        $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
        $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
        attr-dict
    }];
}

def EmitPythonBytecode_Compare: EmitPythonBytecode_Op<"COMPARE"> {
    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs, UI8Attr:$predicate);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadAssertionError : EmitPythonBytecode_Op<"LOAD_ASSERTION_ERROR"> {
    let summary = [{
Loads AssertionError type. This is guaranteed to be the builtin
AssertionError.
    }];

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_RaiseVarargs : EmitPythonBytecode_Op<"RAISE_VARARGS", [Terminator]> {
    let summary = "Raises the exception";

    let successors = (successor VariadicSuccessor<AnySuccessor>:$handler);

    let arguments = (ins Python_PyObjectType:$exception,
                         Optional<Python_PyObjectType>:$cause);}

def EmitPythonBytecode_ReRaiseOp: EmitPythonBytecode_Op<"RERAISE", [Terminator]> {

    let successors = (successor VariadicSuccessor<AnySuccessor>:$handler);

    let summary = "Re-raise the last exception";
}

def EmitPythonBytecode_UnaryOp : EmitPythonBytecode_Op<"UNARY"> {
    let summary = "Unary operation";

    let arguments = (ins Python_PyObjectType:$input, UI8Attr:$operation_type);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_BuildDict : EmitPythonBytecode_Op<"BUILD_DICT", [SameVariadicOperandSize]> {
    let summary = "Build a dictionary object using keys and values";

    let arguments = (ins Variadic<Python_PyObjectType>:$keys,
                         Variadic<Python_PyObjectType>:$values);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_DictUpdate : EmitPythonBytecode_Op<"DICT_UPDATE"> {
    let summary = "Updates a dictionary's entries using another mappable type";

    let arguments = (ins Python_PyObjectType:$dict,
                         Python_PyObjectType:$mappable);
}

def EmitPythonBytecode_DictAdd : EmitPythonBytecode_Op<"DICT_ADD"> {
    let summary = "Adds a key value pair to a dictionary";

    let arguments = (ins Python_PyObjectType:$dict,
                         Python_PyObjectType:$key,
                         Python_PyObjectType:$value);
}

def EmitPythonBytecode_BuildList : EmitPythonBytecode_Op<"BUILD_LIST"> {
    let summary = "Builds a Python list";

    let arguments = (ins Variadic<Python_PyObjectType>:$elements);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_ListExtend : EmitPythonBytecode_Op<"LIST_EXTEND"> {
    let summary = "Extends a Python list";

    let arguments = (ins Python_PyObjectType:$list, Python_PyObjectType:$iterable);
}

def EmitPythonBytecode_ListAppend : EmitPythonBytecode_Op<"LIST_APPEND"> {
    let summary = "Append to a Python list";

    let arguments = (ins Python_PyObjectType:$list, Python_PyObjectType:$value);
}

def EmitPythonBytecode_ListToTuple : EmitPythonBytecode_Op<"LIST_TO_TUPLE"> {
    let summary = "Builds a Python tuple from a list";

    let arguments = (ins Python_PyObjectType:$list);

    let results = (outs Python_PyObjectType:$tuple);
}

def EmitPythonBytecode_BuildTuple : EmitPythonBytecode_Op<"BUILD_TUPLE"> {
    let summary = "Builds a Python tuple";

    let arguments = (ins Variadic<Python_PyObjectType>:$elements);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_BuildSet : EmitPythonBytecode_Op<"BUILD_SET"> {
    let summary = "Builds a Python set";

    let arguments = (ins Variadic<Python_PyObjectType>:$elements);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_SetAdd : EmitPythonBytecode_Op<"SET_ADD"> {
    let summary = "Adds elemets to a set";

    let arguments = (ins Python_PyObjectType:$set, Python_PyObjectType:$element);
}

def EmitPythonBytecode_SetUpdate : EmitPythonBytecode_Op<"SET_UPDATE"> {
    let summary = "Updates a set with an iterable";

    let arguments = (ins Python_PyObjectType:$set, Python_PyObjectType:$iterable);
}

def EmitPythonBytecode_BuildString : EmitPythonBytecode_Op<"BUILD_STRING"> {
    let summary = "Build a string";

    let arguments = (ins Variadic<Python_PyObjectType>:$elements);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_BuildSlice: EmitPythonBytecode_Op<"BUILD_SLICE"> {
    let summary = "Build a slice object";

    let arguments = (ins Python_PyObjectType:$lower,
                         Python_PyObjectType:$upper,
                         Python_PyObjectType:$step);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_FormatValue : EmitPythonBytecode_Op<"FORMAT_VALUE"> {
    let summary = "Format value as a string using specified conversion";

    let arguments = (ins Python_PyObjectType:$value,
                         UI8Attr:$conversion);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadAttribute: EmitPythonBytecode_Op<"LOAD_ATTRIBUTE"> {
    let summary = "Load an attribute";

    let arguments = (ins Python_PyObjectType:$self,
                         StrAttr:$attr);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_DeleteAttribute: EmitPythonBytecode_Op<"DELETE_ATTRIBUTE"> {
    let summary = "Delete an attribute";

    let arguments = (ins Python_PyObjectType:$self,
                         StrAttr:$attribute);
}

def EmitPythonBytecode_BinarySubscript: EmitPythonBytecode_Op<"BINARY_SUBSCRIPT"> {
    let summary = "Subscript an object";

    let arguments = (ins Python_PyObjectType:$self,
                         Python_PyObjectType:$subscript);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_StoreSubscript: EmitPythonBytecode_Op<"STORE_SUBSCRIPT"> {
    let summary = "Store value in object using a subscript";

    let arguments = (ins Python_PyObjectType:$self,
                         Python_PyObjectType:$subscript,
                         Python_PyObjectType:$value);
}

def EmitPythonBytecode_DeleteSubscript: EmitPythonBytecode_Op<"DELETE_SUBSCRIPT"> {
    let summary = "Delete value in object using a subscript";

    let arguments = (ins Python_PyObjectType:$self,
                         Python_PyObjectType:$subscript);
}

def EmitPythonBytecode_StoreAttribute: EmitPythonBytecode_Op<"STORE_ATTRIBUTE"> {
    let summary = "Store value in object using an attribute";

    let arguments = (ins Python_PyObjectType:$self,
                         StrAttr:$attribute,
                         Python_PyObjectType:$value);
}

def EmitPythonBytecode_LoadMethod: EmitPythonBytecode_Op<"LOAD_METHOD"> {
    let summary = "Load a method";

    let arguments = (ins Python_PyObjectType:$self,
                         StrAttr:$method_name);

    let results = (outs Python_PyObjectType:$method);
}

def EmitPythonBytecode_MakeFunction : EmitPythonBytecode_Op<"MAKE_FUNCTION", [AttrSizedOperandSegments]> {
    let summary = "Instantiates a new function object";

    let arguments = (ins Python_PyObjectType:$sym_name,
                         Variadic<Python_PyObjectType>:$defaults,
                         Variadic<Python_PyObjectType>:$kw_defaults,
                         Optional<Python_PyObjectType>:$captures);
    let results = (outs Python_PyObjectType:$func_object);
}

def EmitPythonBytecode_Push : EmitPythonBytecode_Op<"PUSH"> {
    let summary = "Push register value to the stack";

    let arguments = (ins UI64Attr:$src);
}

def EmitPythonBytecode_Pop : EmitPythonBytecode_Op<"POP"> {
    let summary = "Pop the top of the stack to register";

    let arguments = (ins UI64Attr:$dst);
}

def EmitPythonBytecode_Move : EmitPythonBytecode_Op<"MOVE"> {
    let summary = "Move register value to another register";

    let arguments = (ins UI64Attr:$dst, UI64Attr:$src);
}

def EmitPythonBytecode_SetupExceptionHandle : EmitPythonBytecode_Op<"SETUP_EXC_HANDLE", [Terminator]> {
    let summary = "Setup exception handling";

    let successors = (successor AnySuccessor:$body, AnySuccessor:$handler);
}

def EmitPythonBytecode_SetupWith : EmitPythonBytecode_Op<"SETUP_WITH", [Terminator]> {
    let summary = "Setup with handling";

    let successors = (successor AnySuccessor:$body, AnySuccessor:$handler);
}

def EmitPythonBytecode_WithExceptStart : EmitPythonBytecode_Op<"WITH_EXCEPT_START"> {
    let arguments = (ins Python_PyObjectType:$exit_method);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_ClearExceptionState: EmitPythonBytecode_Op<"CLEAR_EXCEPTION_STATE"> {
    let summary = "Clears the interpreter exception state";
}

def EmitPythonBytecode_LeaveExceptionHandle : EmitPythonBytecode_Op<"LEAVE_EXCEPTION_HANDLE"> {
    let summary = "Leave current exception handling";
}

def EmitPythonBytecode_ImportName : EmitPythonBytecode_Op<"IMPORT_NAME"> {
    let summary = "Import Python module";

    let arguments = (ins StrAttr:$name,
                         Python_PyObjectType:$level,
                         Python_PyObjectType:$from_list);

    let results = (outs Python_PyObjectType:$module);
}

def EmitPythonBytecode_ImportFrom : EmitPythonBytecode_Op<"IMPORT_FROM"> {
    let summary = "Import an object from a Python module using its name";

    let arguments = (ins Python_PyObjectType:$module,
                         StrAttr:$name);

    let results = (outs Python_PyObjectType:$object);
}

def EmitPythonBytecode_ImportAll : EmitPythonBytecode_Op<"IMPORT_ALL"> {
    let summary = "Import all objects from a Python module";

    let arguments = (ins Python_PyObjectType:$module);
}

def EmitPythonBytecode_GetIter: EmitPythonBytecode_Op<"GET_ITER"> {
    let summary = "Convenience op that calls Python's iter function on an iterable";

    let arguments = (ins Python_PyObjectType:$iterable);

    let results = (outs Python_PyObjectType:$iterator);
}

def EmitPythonBytecode_ForIter: EmitPythonBytecode_Op<"FOR_ITER", [Terminator,
                                                                   DeclareOpInterfaceMethods<BranchOpInterface>]> {
    let summary = "Convenience op that calls Python's next function on an iterator";

    let arguments = (ins Python_PyObjectType:$iterator);

    let successors = (successor AnySuccessor:$body, AnySuccessor:$continuation);
}

def EmitPythonBytecode_CastToBool: EmitPythonBytecode_Op<"TO_BOOL"> {
    let summary = "Converts a value to a Python bool";

    let arguments = (ins Python_PyObjectType:$value);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadBuildClass: EmitPythonBytecode_Op<"LOAD_BUILD_CLASS"> {
    let summary = "Loads the builtin class builder factory function";

    let results = (outs Python_PyObjectType:$class_builder);
}

def EmitPythonBytecode_Yield : EmitPythonBytecode_Op<"YIELD_VALUE"> {
    let summary = "Yield value from generator";

    let arguments = (ins Python_PyObjectType:$value);

    let results = (outs Python_PyObjectType:$received);
}

def EmitPythonBytecode_YieldFromIter : EmitPythonBytecode_Op<"YIELD_FROM_ITER"> {
    let summary = "Get the iterator from iterable with special cases for generators and coroutines";

    let arguments = (ins Python_PyObjectType:$iterable);

    let results = (outs Python_PyObjectType:$iterator);
}

def EmitPythonBytecode_YieldFrom : EmitPythonBytecode_Op<"YIELD_FROM"> {
    let summary = "Yield iterator from generator";

    let arguments = (ins Python_PyObjectType:$iterator, Python_PyObjectType:$value);

    let results = (outs Python_PyObjectType:$received);
}

def UnpackSequenceOp: EmitPythonBytecode_Op<"UNPACK_SEQUENCE"> {
    let summary = "Unpack sequence";

    let arguments = (ins Python_PyObjectType:$iterable);

    let results = (outs Variadic<Python_PyObjectType>:$unpacked_values);
}

def UnpackExpandOp: EmitPythonBytecode_Op<"UNPACK_EXPAND"> {
    let summary = "Unpack and expand rest";

    let arguments = (ins Python_PyObjectType:$iterable);

    let results = (outs Variadic<Python_PyObjectType>:$unpacked_values,
                        Python_PyObjectType:$rest);
}

def GetAwaitableOp : EmitPythonBytecode_Op<"GET_AWAITABLE"> {
    let summary = "Get awaitable";

    let arguments = (ins Python_PyObjectType:$iterable);

    let results = (outs Python_PyObjectType:$iterator);
}