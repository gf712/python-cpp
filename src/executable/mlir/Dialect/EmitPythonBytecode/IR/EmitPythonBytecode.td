include "EmitPythonBytecodeBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "Python/IR/PythonTypes.td"

// Base class for EmitPythonBytecode dialect ops.
class EmitPythonBytecode_Op<string mnemonic, list<Trait> traits = []>
    : Op<EmitPythonBytecode_Dialect, mnemonic, traits>;

def EmitPythonBytecode_ConstantOp: EmitPythonBytecode_Op<"LOAD_CONST", [Pure]> {
    let arguments = (ins AnyAttr:$value);

    let results = (outs Python_PyObjectType:$output);

    let hasVerifier = 1;
}

def EmitPythonBytecode_LoadEllipsisOp: EmitPythonBytecode_Op<"LOAD_ELLIPSIS", [Pure]> {
    let results = (outs Python_PyObjectType:$output);
}

// LOAD_* ops resolve a name in the local / module / global / cell
// environment — they're *readers* of the binding state. Marking them
// MemRead on DefaultResource is what hooks them up to RDV's per-
// resource liveness analysis: a STORE_NAME's MemWrite to the same
// resource is then kept alive because a reachable reader exists.
// (The matching may-raise NameError / UnboundLocalError side effect
// is modelled separately on the source dialect via the
// PythonExceptionStateResource; this dialect runs post-conversion
// where the bytecode emitter handles exception state via its own
// resource-less instructions.)
def EmitPythonBytecode_LoadNameOp: EmitPythonBytecode_Op<"LOAD_NAME",
        [MemoryEffects<[MemRead]>]> {
    let arguments = (ins StrAttr:$name);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadFastOp: EmitPythonBytecode_Op<"LOAD_FAST",
        [MemoryEffects<[MemRead]>]> {
    let arguments = (ins StrAttr:$name);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadGlobalOp: EmitPythonBytecode_Op<"LOAD_GLOBAL",
        [MemoryEffects<[MemRead]>]> {
    let arguments = (ins StrAttr:$name);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadDerefOp: EmitPythonBytecode_Op<"LOAD_DEREF",
        [MemoryEffects<[MemRead]>]> {
    let arguments = (ins StrAttr:$name);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadClosureOp: EmitPythonBytecode_Op<"LOAD_CLOSURE",
        [MemoryEffects<[MemRead]>]> {
    let arguments = (ins StrAttr:$name);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_StoreNameOp: EmitPythonBytecode_Op<"STORE_NAME", [MemoryEffects<[MemWrite]>]> {
    let arguments = (ins StrAttr:$name, Python_PyObjectType:$object);
}

def EmitPythonBytecode_StoreGlobalOp: EmitPythonBytecode_Op<"STORE_GLOBAL", [MemoryEffects<[MemWrite]>]> {
    let arguments = (ins StrAttr:$name, Python_PyObjectType:$object);
}

def EmitPythonBytecode_StoreFastOp: EmitPythonBytecode_Op<"STORE_FAST", [MemoryEffects<[MemWrite]>]> {
    let arguments = (ins StrAttr:$name, Python_PyObjectType:$object);
}

def EmitPythonBytecode_StoreDerefOp: EmitPythonBytecode_Op<"STORE_DEREF", [MemoryEffects<[MemWrite]>]> {
    let arguments = (ins StrAttr:$name, Python_PyObjectType:$object);
}

def EmitPythonBytecode_DeleteNameOp: EmitPythonBytecode_Op<"DELETE_NAME", [MemoryEffects<[MemWrite]>]> {
    let arguments = (ins StrAttr:$name);
}

def EmitPythonBytecode_DeleteFastOp: EmitPythonBytecode_Op<"DELETE_FAST", [MemoryEffects<[MemWrite]>]> {
    let arguments = (ins StrAttr:$name);
}

def EmitPythonBytecode_DeleteGlobalOp: EmitPythonBytecode_Op<"DELETE_GLOBAL", [MemoryEffects<[MemWrite]>]> {
    let arguments = (ins StrAttr:$name);
}

def EmitPythonBytecode_DeleteDerefOp: EmitPythonBytecode_Op<"DELETE_DEREF", [MemoryEffects<[MemWrite]>]> {
    let arguments = (ins StrAttr:$name);
}

// CALL / CALL_EXPAND / CALL_KW / BINARY_OP / INPLACE_OP / UNARY /
// COMPARE / TO_BOOL / FORMAT_VALUE all dispatch to user-overridable
// dunders or run arbitrary user code. MemWrite is the conservative
// effect: it keeps RemoveDeadValues from stripping a call whose
// result happens to be unused (the user code's side effects are
// still observable).
def EmitPythonBytecode_FunctionCallOp: EmitPythonBytecode_Op<"CALL",
        [MemoryEffects<[MemWrite]>]> {
    let arguments = (ins Python_PyObjectType:$callee, Variadic<Python_PyObjectType>:$args);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_FunctionCallExOp: EmitPythonBytecode_Op<"CALL_EXPAND",
        [AttrSizedOperandSegments, MemoryEffects<[MemWrite]>]> {
    let arguments = (ins Python_PyObjectType:$callee,
                         Optional<Python_PyObjectType>:$args,
                         Optional<Python_PyObjectType>:$kwargs);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_FunctionCallWithKeywordsOp: EmitPythonBytecode_Op<"CALL_KW",
        [AttrSizedOperandSegments, MemoryEffects<[MemWrite]>]> {
    let arguments = (ins Python_PyObjectType:$callee,
                         Variadic<Python_PyObjectType>:$args,
                         Builtin_DenseStringElementsAttr:$keywords,
                         Variadic<Python_PyObjectType>:$kwargs);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_BinaryOp: EmitPythonBytecode_Op<"BINARY_OP",
        [MemoryEffects<[MemWrite]>]> {
    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs, UI8Attr:$operation_type);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_InplaceOp: EmitPythonBytecode_Op<"INPLACE_OP",
        [MemoryEffects<[MemWrite]>]> {
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

def EmitPythonBytecode_Compare: EmitPythonBytecode_Op<"COMPARE",
        [MemoryEffects<[MemWrite]>]> {
    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs, UI8Attr:$predicate);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadAssertionError : EmitPythonBytecode_Op<"LOAD_ASSERTION_ERROR", [Pure]> {
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

def EmitPythonBytecode_UnaryOp : EmitPythonBytecode_Op<"UNARY",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Unary operation";

    let arguments = (ins Python_PyObjectType:$input, UI8Attr:$operation_type);

    let results = (outs Python_PyObjectType:$output);
}

// Build / mutate / coerce collection ops: each constructs an object
// with distinct Python identity or mutates an existing object's
// state. MemWrite documents both the "different invocations produce
// different identities" property (so CSE can't merge two BuildList
// calls) and the explicit-mutation property (DICT_ADD etc.).
def EmitPythonBytecode_BuildDict : EmitPythonBytecode_Op<"BUILD_DICT",
        [SameVariadicOperandSize, MemoryEffects<[MemWrite]>]> {
    let summary = "Build a dictionary object using keys and values";

    let arguments = (ins Variadic<Python_PyObjectType>:$keys,
                         Variadic<Python_PyObjectType>:$values);

    let results = (outs Python_PyObjectType:$output);

    // Register-pressure relief for large dict literals. See
    // ExpandLargeBuildDict in EmitPythonBytecode.cpp.
    let hasCanonicalizer = 1;
}

def EmitPythonBytecode_DictUpdate : EmitPythonBytecode_Op<"DICT_UPDATE",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Updates a dictionary's entries using another mappable type";

    let arguments = (ins Python_PyObjectType:$dict,
                         Python_PyObjectType:$mappable);
}

def EmitPythonBytecode_DictAdd : EmitPythonBytecode_Op<"DICT_ADD",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Adds a key value pair to a dictionary";

    let arguments = (ins Python_PyObjectType:$dict,
                         Python_PyObjectType:$key,
                         Python_PyObjectType:$value);
}

def EmitPythonBytecode_BuildList : EmitPythonBytecode_Op<"BUILD_LIST",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Builds a Python list";

    let arguments = (ins Variadic<Python_PyObjectType>:$elements);

    let results = (outs Python_PyObjectType:$output);

    // Register-pressure relief for all-constants list literals. See
    // FoldAllConstBuildListIntoExtend in EmitPythonBytecode.cpp.
    let hasCanonicalizer = 1;
}

def EmitPythonBytecode_ListExtend : EmitPythonBytecode_Op<"LIST_EXTEND",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Extends a Python list";

    let arguments = (ins Python_PyObjectType:$list, Python_PyObjectType:$iterable);
}

def EmitPythonBytecode_ListAppend : EmitPythonBytecode_Op<"LIST_APPEND",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Append to a Python list";

    let arguments = (ins Python_PyObjectType:$list, Python_PyObjectType:$value);
}

def EmitPythonBytecode_ListToTuple : EmitPythonBytecode_Op<"LIST_TO_TUPLE",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Builds a Python tuple from a list";

    let arguments = (ins Python_PyObjectType:$list);

    let results = (outs Python_PyObjectType:$tuple);
}

def EmitPythonBytecode_BuildTuple : EmitPythonBytecode_Op<"BUILD_TUPLE",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Builds a Python tuple";

    let arguments = (ins Variadic<Python_PyObjectType>:$elements);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_BuildSet : EmitPythonBytecode_Op<"BUILD_SET",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Builds a Python set";

    let arguments = (ins Variadic<Python_PyObjectType>:$elements);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_SetAdd : EmitPythonBytecode_Op<"SET_ADD",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Adds an element to a set";

    let arguments = (ins Python_PyObjectType:$set, Python_PyObjectType:$value);
}

def EmitPythonBytecode_SetUpdate : EmitPythonBytecode_Op<"SET_UPDATE",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Updates a set with an iterable";

    let arguments = (ins Python_PyObjectType:$set, Python_PyObjectType:$iterable);
}

def EmitPythonBytecode_BuildString : EmitPythonBytecode_Op<"BUILD_STRING",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Build a string";

    let arguments = (ins Variadic<Python_PyObjectType>:$elements);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_BuildSlice: EmitPythonBytecode_Op<"BUILD_SLICE",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Build a slice object";

    // `step` is optional: matches py.build_slice in the source dialect,
    // and the bytecode emitter selects the 3- vs 4-arg BuildSlice
    // instruction constructor based on whether step is present.
    let arguments = (ins Python_PyObjectType:$lower,
                         Python_PyObjectType:$upper,
                         Optional<Python_PyObjectType>:$step);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_FormatValue : EmitPythonBytecode_Op<"FORMAT_VALUE",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Format value as a string using specified conversion";

    let arguments = (ins Python_PyObjectType:$value,
                         UI8Attr:$conversion);

    let results = (outs Python_PyObjectType:$output);
}

// Attribute / method / subscript access: each dispatches to a
// user-overridable __getattr__ / __getitem__ / __setitem__ /
// __setattr__ / __delitem__ / __delattr__.
def EmitPythonBytecode_LoadAttribute: EmitPythonBytecode_Op<"LOAD_ATTRIBUTE",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Load an attribute";

    let arguments = (ins Python_PyObjectType:$self,
                         StrAttr:$attr);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_DeleteAttribute: EmitPythonBytecode_Op<"DELETE_ATTRIBUTE",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Delete an attribute";

    let arguments = (ins Python_PyObjectType:$self,
                         StrAttr:$attr);
}

def EmitPythonBytecode_BinarySubscript: EmitPythonBytecode_Op<"BINARY_SUBSCRIPT",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Subscript an object";

    let arguments = (ins Python_PyObjectType:$self,
                         Python_PyObjectType:$subscript);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_StoreSubscript: EmitPythonBytecode_Op<"STORE_SUBSCRIPT",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Store value in object using a subscript";

    let arguments = (ins Python_PyObjectType:$self,
                         Python_PyObjectType:$subscript,
                         Python_PyObjectType:$value);
}

def EmitPythonBytecode_DeleteSubscript: EmitPythonBytecode_Op<"DELETE_SUBSCRIPT",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Delete value in object using a subscript";

    let arguments = (ins Python_PyObjectType:$self,
                         Python_PyObjectType:$subscript);
}

def EmitPythonBytecode_StoreAttribute: EmitPythonBytecode_Op<"STORE_ATTRIBUTE",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Store value in object using an attribute";

    let arguments = (ins Python_PyObjectType:$self,
                         StrAttr:$attr,
                         Python_PyObjectType:$value);
}

def EmitPythonBytecode_LoadMethod: EmitPythonBytecode_Op<"LOAD_METHOD",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Load a method";

    let arguments = (ins Python_PyObjectType:$self,
                         StrAttr:$method_name);

    let results = (outs Python_PyObjectType:$method);
}

def EmitPythonBytecode_MakeFunction : EmitPythonBytecode_Op<"MAKE_FUNCTION",
        [AttrSizedOperandSegments, MemoryEffects<[MemWrite]>]> {
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

def EmitPythonBytecode_WithExceptStart : EmitPythonBytecode_Op<"WITH_EXCEPT_START",
        [MemoryEffects<[MemWrite]>]> {
    let arguments = (ins Python_PyObjectType:$exit_method);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_ClearExceptionState: EmitPythonBytecode_Op<"CLEAR_EXCEPTION_STATE",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Clears the interpreter exception state";
}

def EmitPythonBytecode_LeaveExceptionHandle : EmitPythonBytecode_Op<"LEAVE_EXCEPTION_HANDLE",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Leave current exception handling";
}

def EmitPythonBytecode_ImportName : EmitPythonBytecode_Op<"IMPORT_NAME",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Import Python module";

    let arguments = (ins StrAttr:$name,
                         Python_PyObjectType:$level,
                         Python_PyObjectType:$from_list);

    let results = (outs Python_PyObjectType:$module);
}

def EmitPythonBytecode_ImportFrom : EmitPythonBytecode_Op<"IMPORT_FROM",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Import an object from a Python module using its name";

    let arguments = (ins Python_PyObjectType:$module,
                         StrAttr:$name);

    let results = (outs Python_PyObjectType:$object);
}

def EmitPythonBytecode_ImportAll : EmitPythonBytecode_Op<"IMPORT_ALL",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Import all objects from a Python module";

    let arguments = (ins Python_PyObjectType:$module);
}

def EmitPythonBytecode_GetIter: EmitPythonBytecode_Op<"GET_ITER",
        [MemoryEffects<[MemWrite]>]> {
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

def EmitPythonBytecode_CastToBool: EmitPythonBytecode_Op<"TO_BOOL",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Converts a value to a Python bool";

    let arguments = (ins Python_PyObjectType:$value);

    let results = (outs Python_PyObjectType:$output);
}

def EmitPythonBytecode_LoadBuildClass: EmitPythonBytecode_Op<"LOAD_BUILD_CLASS", [Pure]> {
    let summary = "Loads the builtin class builder factory function";

    let results = (outs Python_PyObjectType:$class_builder);
}

def EmitPythonBytecode_Yield : EmitPythonBytecode_Op<"YIELD_VALUE",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Yield value from generator";

    let arguments = (ins Python_PyObjectType:$value);

    let results = (outs Python_PyObjectType:$received);
}

def EmitPythonBytecode_YieldFromIter : EmitPythonBytecode_Op<"YIELD_FROM_ITER",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Get the iterator from iterable with special cases for generators and coroutines";

    let arguments = (ins Python_PyObjectType:$iterable);

    let results = (outs Python_PyObjectType:$iterator);
}

def EmitPythonBytecode_YieldFrom : EmitPythonBytecode_Op<"YIELD_FROM",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Yield iterator from generator";

    let arguments = (ins Python_PyObjectType:$iterator, Python_PyObjectType:$value);

    let results = (outs Python_PyObjectType:$received);
}

def UnpackSequenceOp: EmitPythonBytecode_Op<"UNPACK_SEQUENCE",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Unpack sequence";

    let arguments = (ins Python_PyObjectType:$iterable);

    let results = (outs Variadic<Python_PyObjectType>:$unpacked_values);
}

def UnpackExpandOp: EmitPythonBytecode_Op<"UNPACK_EXPAND",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Unpack and expand rest";

    let arguments = (ins Python_PyObjectType:$iterable);

    let results = (outs Variadic<Python_PyObjectType>:$unpacked_values,
                        Python_PyObjectType:$rest);
}

def GetAwaitableOp : EmitPythonBytecode_Op<"GET_AWAITABLE",
        [MemoryEffects<[MemWrite]>]> {
    let summary = "Get awaitable";

    let arguments = (ins Python_PyObjectType:$iterable);

    let results = (outs Python_PyObjectType:$iterator);
}
