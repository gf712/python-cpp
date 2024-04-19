include "Python/IR/PythonAttributes.td"
include "Python/IR/PythonTypes.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

class Python_Op<string mneomic, list<Trait> traits = []> : Op<Python_Dialect, mneomic, traits>;

def ConstantOp : Python_Op<"constant"> {
    let summary = "Build a PyObject from a constant";

    let arguments = (ins AnyAttr:$value);

    let results = (outs Python_PyObjectType:$output);

    let builders = [
        OpBuilder<(ins "double":$value)>,
        OpBuilder<(ins "IntegerAttr":$value)>,
        OpBuilder<(ins "bool":$value)>,
        OpBuilder<(ins "NoneType":$value)>,
        OpBuilder<(ins "StringAttr":$value)>,
        OpBuilder<(ins "std::vector<std::byte>":$value)>,
        OpBuilder<(ins "::mlir::py::PyEllipsisType":$value)>,
    ];
}

def LoadNameOp : Python_Op<"load_name"> {
    let summary = "Load PyObject from the local environment";

    let arguments = (ins StrAttr:$name);

    let results = (outs Python_PyObjectType:$output);
}

def LoadFastOp : Python_Op<"load_fast"> {
    let summary = "Load PyObject bound variable";

    let arguments = (ins StrAttr:$name);

    let results = (outs Python_PyObjectType:$output);
}

def LoadGlobalOp : Python_Op<"load_global"> {
    let summary = "Load PyObject from the global environment";

    let arguments = (ins StrAttr:$name);

    let results = (outs Python_PyObjectType:$output);
}

def LoadDerefOp : Python_Op<"load_deref"> {
    let summary = "Load PyObject from a cell";

    let arguments = (ins StrAttr:$name);

    let results = (outs Python_PyObjectType:$output);
}

def StoreNameOp : Python_Op<"store_name"> {
    let summary = "Store value to the local environment";

    let arguments = (ins StrAttr:$name, Python_PyObjectType:$value);

    let results = (outs Python_PyObjectType:$output);
}

def StoreFastOp : Python_Op<"store_fast"> {
    let summary = "Store value to local bound variable";

    let arguments = (ins StrAttr:$name, Python_PyObjectType:$value);

    let results = (outs Python_PyObjectType:$output);
}

def StoreGlobalOp : Python_Op<"store_global"> {
    let summary = "Store value in the global environment";

    let arguments = (ins StrAttr:$name, Python_PyObjectType:$value);

    let results = (outs Python_PyObjectType:$output);
}

def StoreDerefOp : Python_Op<"store_deref"> {
    let summary = "Store PyObject to a cell";

    let arguments = (ins StrAttr:$name, Python_PyObjectType:$value);

    let results = (outs Python_PyObjectType:$output);
}

def DeleteNameOp : Python_Op<"delete_name"> {
    let summary = "Delete value from the local environment";

    let arguments = (ins StrAttr:$name);
}

def DeleteFastOp : Python_Op<"delete_fast"> {
    let summary = "Delete value from the bound variable";

    let arguments = (ins StrAttr:$name);
}

def DeleteGlobalOp : Python_Op<"delete_global"> {
    let summary = "Delete value from the global environment";

    let arguments = (ins StrAttr:$name);
}

def MakeFunctionOp : Python_Op<"make_function", [AttrSizedOperandSegments]> {
    let summary = "Make a function using the local environment and code object";

    let arguments = (ins FlatSymbolRefAttr:$function_name,
                         Variadic<Python_PyObjectType>:$defaults,
                         Variadic<Python_PyObjectType>:$kw_defaults,
                         Builtin_DenseStringElementsAttr:$captures);

    let results = (outs Python_PyObjectType:$func_object);
}

def FunctionCallOp : Python_Op<"call", [AttrSizedOperandSegments]> {
    let summary = "A generic function call";

    let arguments = (ins Python_PyObjectType:$callee,
                         Variadic<Python_PyObjectType>:$args,
                         Builtin_DenseStringElementsAttr:$keywords,
                         Variadic<Python_PyObjectType>:$kwargs,
                         BoolAttr:$requires_args_expansion,
                         BoolAttr:$requires_kwargs_expansion
                         );

    let results = (outs Python_PyObjectType:$output);
}

def BinaryAddOp : Python_Op<"add"> {
    let summary = "Generic binary addition operation";

    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs);

    let results = (outs Python_PyObjectType:$output);
}

def BinarySubtractOp : Python_Op<"sub"> {
    let summary = "Generic binary subtraction operation";

    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs);

    let results = (outs Python_PyObjectType:$output);
}

def BinaryModuloOp : Python_Op<"mod"> {
    let summary = "Generic binary modulo operation";

    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs);

    let results = (outs Python_PyObjectType:$output);
}

def BinaryMultiplyOp : Python_Op<"mul"> {
    let summary = "Generic binary multiplication operation";

    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs);

    let results = (outs Python_PyObjectType:$output);
}

def BinaryExpOp : Python_Op<"exp"> {
    let summary = "Generic binary exponential operation";

    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs);

    let results = (outs Python_PyObjectType:$output);
}

def BinaryDivOp : Python_Op<"div"> {
    let summary = "Generic binary division operation";

    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs);

    let results = (outs Python_PyObjectType:$output);
}

def BinaryFloorDivOp : Python_Op<"floordiv"> {
    let summary = "Generic binary floor division operation";

    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs);

    let results = (outs Python_PyObjectType:$output);
}

def BinaryMatMulOp : Python_Op<"matmul"> {
    let summary = "Generic binary matrix multiplication operation";

    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs);

    let results = (outs Python_PyObjectType:$output);
}

def LeftShiftOp : Python_Op<"lshift"> {
    let summary = "Generic binary left shift operation";

    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs);

    let results = (outs Python_PyObjectType:$output);
}

def RightShiftOp : Python_Op<"rshift"> {
    let summary = "Generic binary right shift operation";

    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs);

    let results = (outs Python_PyObjectType:$output);
}

def LogicalAndOp : Python_Op<"logical_and"> {
    let summary = "Generic binary logical and operation";

    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs);

    let results = (outs Python_PyObjectType:$output);
}

def LogicalOrOp : Python_Op<"logical_or"> {
    let summary = "Generic binary logical or operation";

    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs);

    let results = (outs Python_PyObjectType:$output);
}

def LogicalXorOp : Python_Op<"logical_xor"> {
    let summary = "Generic binary logical xor operation";

    let arguments = (ins Python_PyObjectType:$lhs, Python_PyObjectType:$rhs);

    let results = (outs Python_PyObjectType:$output);
}

def InplaceOp : Python_Op<"inplace_op"> {
    let summary = "Generic inplace operation";

    let arguments = (ins Python_PyObjectType:$src,
                         Python_PyObjectType:$dst,
                         Python_InplaceOpKindAttr:$kind);

    let results = (outs Python_PyObjectType:$result);
}

def PositiveOp : Python_Op<"pos"> {
    let summary = "Unary positive operation";

    let arguments = (ins Python_PyObjectType:$input);

    let results = (outs Python_PyObjectType:$output);
}

def NegativeOp : Python_Op<"neg"> {
    let summary = "Unary negative operation";

    let arguments = (ins Python_PyObjectType:$input);

    let results = (outs Python_PyObjectType:$output);
}

def InvertOp : Python_Op<"inv"> {
    let summary = "Unary invert operation";

    let arguments = (ins Python_PyObjectType:$input);

    let results = (outs Python_PyObjectType:$output);
}

def NotOp : Python_Op<"not"> {
    let summary = "Unary negation operation";

    let arguments = (ins Python_PyObjectType:$input);

    let results = (outs Python_PyObjectType:$output);
}

def CastToBoolOp : Python_Op<"as_bool"> {
    let summary = "Convert Python object to bool";

    let arguments = (ins Python_PyObjectType:$value);

    let results = (outs I1:$output);
}

def Compare : Python_Op<"cmp"> {
    let summary = "Compare two operands";

    let arguments = (ins Python_CmpPredicateAttr:$predicate,
                         Python_PyObjectType:$lhs,
                         Python_PyObjectType:$rhs);

    let results = (outs Python_PyObjectType:$output);
}

def LoadAssertionError : Python_Op<"load_assertion_error"> {
    let summary = [{
Loads AssertionError type. This is guaranteed to be the builtin
AssertionError.
    }];

    let results = (outs Python_PyObjectType:$output);
}

def BuildDictOp : Python_Op<"build_dict", [SameVariadicOperandSize]> {
    let summary = "Builds a literal dictionary and expands the kv pairs as needed";

    let arguments = (ins Variadic<Python_PyObjectType>:$keys,
                         Variadic<Python_PyObjectType>:$values,
                         DenseBoolArrayAttr:$requires_expansion);

    let results = (outs Python_PyObjectType:$output);
}

def DictAddOp : Python_Op<"dict_add"> {
    let summary = "Adds a key/value pair to the provided dictionary";

    let arguments = (ins Python_PyObjectType:$dict,
                         Python_PyObjectType:$key,
                         Python_PyObjectType:$value);
}

def BuildListOp : Python_Op<"build_list"> {
    let summary = "Builds a literal Python list and expands elements where needed";

    let arguments = (ins Variadic<Python_PyObjectType>:$elements,
                         DenseBoolArrayAttr:$requires_expansion);

    let results = (outs Python_PyObjectType:$output);
}

def ListAppendOp : Python_Op<"list_append"> {
    let summary = "Append to a Python list";

    let arguments = (ins Python_PyObjectType:$list, Python_PyObjectType:$value);
}

def BuildTupleOp : Python_Op<"build_tuple"> {
    let summary = "Builds a literal Python tuple and expands elements where needed";

    let arguments = (ins Variadic<Python_PyObjectType>:$elements,
                         DenseBoolArrayAttr:$requires_expansion);

    let results = (outs Python_PyObjectType:$output);
}

def BuildSetOp : Python_Op<"build_set"> {
    let summary = "Builds a literal Python set and expands elements where needed";

    let arguments = (ins Variadic<Python_PyObjectType>:$elements,
                         DenseBoolArrayAttr:$requires_expansion);

    let results = (outs Python_PyObjectType:$output);
}

def SetAddOp : Python_Op<"set_add"> {
    let summary = "Add a value to a set";

    let arguments = (ins Python_PyObjectType:$set,
                         Python_PyObjectType:$value);
}

def BuildStringOp : Python_Op<"build_string"> {
    let summary = "Builds a string";

    let arguments = (ins Variadic<Python_PyObjectType>:$elements);

    let results = (outs Python_PyObjectType:$output);
}

def FormatValueOp : Python_Op<"format"> {
    let summary = "Format an object as a string using the requested conversion";

    let arguments = (ins Python_PyObjectType:$value,
                         Python_FormatValueConversionAttr:$conversion);

    let results = (outs Python_PyObjectType:$output);
}

def LoadAttributeOp: Python_Op<"load_attribute"> {
    let summary = "Load an attribute";

    let arguments = (ins Python_PyObjectType:$self,
                         StrAttr:$attr);

    let results = (outs Python_PyObjectType:$output);
}

def BinarySubscriptOp: Python_Op<"subscript"> {
    let summary = "Subscript an object";

    let arguments = (ins Python_PyObjectType:$self,
                         Python_PyObjectType:$subscript);

    let results = (outs Python_PyObjectType:$output);
}

def StoreSubscriptOp: Python_Op<"store_subscript"> {
    let summary = "Store value in object using a subscript";

    let arguments = (ins Python_PyObjectType:$self,
                         Python_PyObjectType:$subscript,
                         Python_PyObjectType:$value);
}

def DeleteSubscriptOp: Python_Op<"delete_subscript"> {
    let summary = "Delete value in object using a subscript";

    let arguments = (ins Python_PyObjectType:$self,
                         Python_PyObjectType:$subscript);
}

def StoreAttributeOp: Python_Op<"store_attribute"> {
    let summary = "Store value in object using an attribute";

    let arguments = (ins Python_PyObjectType:$self,
                         StrAttr:$attribute,
                         Python_PyObjectType:$value);
}

def LoadMethodOp: Python_Op<"load_method"> {
    let summary = "Load a method";

    let arguments = (ins Python_PyObjectType:$self,
                         StrAttr:$method_name);

    let results = (outs Python_PyObjectType:$method);
}

def UnpackSequenceOp: Python_Op<"unpack"> {
    let summary = "Unpack ";

    let arguments = (ins Python_PyObjectType:$iterable);

    let results = (outs Variadic<Python_PyObjectType>:$unpacked_values);
}

def ForIterOp: Python_Op<"for_iter", [Terminator]> {
    let summary = "Convenience op for for loops";

    let description = [{
Jumps to `end` if calling Python's `next` builtin raises StopIteration.
Otherwise, returns the result of `next`.
If an exception that is not a subclass of StopIteration is raised, that
exception is thrown.
    }];

    let arguments = (ins Python_PyObjectType:$iterator);

    let successors = (successor AnySuccessor:$start, AnySuccessor:$end);

    let results = (outs Python_PyObjectType:$value);
}

def ForLoopOp: Python_Op<"for_loop", [DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
    let summary = "For loop representation";

    let arguments = (ins Python_PyObjectType:$iterable);

    let regions = (region MinSizedRegion<1>:$body,
                          SizedRegion<1>:$step,
                          MinSizedRegion<1>:$orelse);
}

def WhileOp: Python_Op<"while", [DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
    let summary = "While loop representation";

    let regions = (region AnyRegion:$condition, AnyRegion:$body, AnyRegion:$orelse);
}

def ConditionOp: Python_Op<"condition", [Terminator,
                                         ParentOneOf<["WhileOp", "TryHandlerScope"]>]> {
    let summary = "Condition of a while loop or a catch statement";

    let arguments = (ins Python_PyObjectType:$cond);
}

def TryHandlerScope: Python_Op<"catch", [Terminator,
                                         HasParent<"TryOp">,
                                         DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
    let regions = (region AnyRegion:$cond,
                          AnyRegion:$handler);
}

def TryOp: Python_Op<"try", [DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
    let regions = (region AnyRegion:$body,
                          AnyRegion:$orelse,
                          AnyRegion:$finally,
                          VariadicRegion<SizedRegion<1>>:$handlers);
}

def WithOp: Python_Op<"with"> {
    let arguments = (ins Variadic<Python_PyObjectType>:$items);

    let regions = (region AnyRegion:$body);
}

def WithExceptStartOp: Python_Op<"with_except_start"> {
    let arguments = (ins Python_PyObjectType:$exit_method);

    let results = (outs Python_PyObjectType:$output);
}

def RaiseOp: Python_Op<"raise", [Terminator,
                                 AttrSizedOperandSegments]> {
    let summary = "Raises an exception";

    let arguments = (ins Optional<Python_PyObjectType>:$exception,
                         Optional<Python_PyObjectType>:$cause);

    let builders = [
        OpBuilder<(ins), [{
            build($_builder, $_state, nullptr, nullptr);
        }]>,
        OpBuilder<(ins "Value":$exception), [{
            build($_builder, $_state, exception, nullptr);
        }]>
    ];
}

def ClearExceptionStateOp: Python_Op<"clear_exc_state"> {
    let summary = "Clears the interpreter exception state";
}

def ControlFlowYield: Python_Op<"cf_yield", [ReturnLike,
                                             DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface, ["getSuccessorRegions"]>,
                                             Terminator,
                                             ParentOneOf<["TryOp",
                                                          "ForLoopOp",
                                                          "WithOp",
                                                          "WhileOp",
                                                          "TryHandlerScope"]>]> {
    let summary = "Yield control to the parent operation";

    let arguments = (ins OptionalAttr<Python_LoopOpKindAttr>:$kind);

    let builders = [
        OpBuilder<(ins), [{
            build($_builder, $_state, nullptr);
        }]>,
    ];

    let description = [{
        This is distinct from Python's `yield`, and it can only be used from within a control flow operation.
    }];
}

def CondBranchSubclassOp: Python_Op<"cond_br_subclass", [Terminator,
    AttrSizedOperandSegments, DeclareOpInterfaceMethods<BranchOpInterface>]> {
    let summary = "Jump if the current active exception is not of `object_type`";

    let arguments = (ins Python_PyObjectType:$object_type,
                         Variadic<Python_PyObjectType>:$trueDestOperands,
                         Variadic<Python_PyObjectType>:$falseDestOperands);

    let successors = (successor AnySuccessor:$trueDest,
                                AnySuccessor:$falseDest);

    let assemblyFormat = [{
        $object_type `,`
        $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
        $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
        attr-dict
    }];
}

def ImportOp : Python_Op<"import"> {
    let summary = "Import a Python module";

    let arguments = (ins StrAttr:$name, Builtin_DenseStringElementsAttr:$from_list, UI32Attr:$level);

    let results = (outs Python_PyObjectType:$module);
}

def ImportFromOp : Python_Op<"import_from"> {
    let summary = "Import an object from a module";

    let arguments = (ins Python_PyObjectType:$module, StrAttr:$name);

    let results = (outs Python_PyObjectType:$object);
}

def ImportAllOp : Python_Op<"import_all"> {
    let summary = "Import all objects from a module into the current scope";

    let arguments = (ins Python_PyObjectType:$module);
}

def ClassDefinitionOp: Python_Op<"class", [AttrSizedOperandSegments]> {
    let summary = "Class definition";

    let arguments = (ins StrAttr:$name,
                         StrAttr:$mangled_name,
                         Variadic<Python_PyObjectType>:$bases,
                         Builtin_DenseStringElementsAttr:$keywords,
                         Variadic<Python_PyObjectType>:$kwargs,
                         Builtin_DenseStringElementsAttr:$captures);

    let regions = (region AnyRegion:$body);

    let results = (outs Python_PyObjectType:$output);
}

def YieldOp : Python_Op<"yield"> {
    let summary = "Yield value from generator";

    let arguments = (ins Python_PyObjectType:$value);

    let results = (outs Python_PyObjectType:$received);
}

def YieldFromOp : Python_Op<"yield_from"> {
    let summary = "Yield iterable from generator";

    let arguments = (ins Python_PyObjectType:$iterable);

    let results = (outs Python_PyObjectType:$received);
}

def BuildSliceOp : Python_Op<"build_slice"> {
    let summary = "Build a slice object from object subscripting";

    let arguments = (ins Python_PyObjectType:$lower,
                         Python_PyObjectType:$upper,
                         Python_PyObjectType:$step);

    let results = (outs Python_PyObjectType:$slice);
}

def GetAwaitableOp : Python_Op<"get_awaitable"> {
    let summary = "Get awaitable";

    let arguments = (ins Python_PyObjectType:$iterable);

    let results = (outs Python_PyObjectType:$iterator);
}