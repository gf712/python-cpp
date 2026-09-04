#pragma once

// The AST node-type X-macro list.
//
// This lives in a plain header rather than in the py.ast module because macros
// are never exported by a module: every consumer that expands AST_NODE_TYPES to
// generate visitor declarations (MLIRGenerator.hpp, VariablesResolver.cppm,
// AST.cpp, ...) has to #include it directly. py.ast includes it in its global
// module fragment for the same reason.

#define AST_NODE_TYPES                       \
	__AST_NODE_TYPE(Argument)                \
	__AST_NODE_TYPE(Arguments)               \
	__AST_NODE_TYPE(Attribute)               \
	__AST_NODE_TYPE(Assign)                  \
	__AST_NODE_TYPE(Assert)                  \
	__AST_NODE_TYPE(AsyncFunctionDefinition) \
	__AST_NODE_TYPE(Await)                   \
	__AST_NODE_TYPE(AugAssign)               \
	__AST_NODE_TYPE(Break)                   \
	__AST_NODE_TYPE(BinaryExpr)              \
	__AST_NODE_TYPE(BoolOp)                  \
	__AST_NODE_TYPE(Call)                    \
	__AST_NODE_TYPE(ClassDefinition)         \
	__AST_NODE_TYPE(Continue)                \
	__AST_NODE_TYPE(Compare)                 \
	__AST_NODE_TYPE(Comprehension)           \
	__AST_NODE_TYPE(Constant)                \
	__AST_NODE_TYPE(Delete)                  \
	__AST_NODE_TYPE(Dict)                    \
	__AST_NODE_TYPE(DictComp)                \
	__AST_NODE_TYPE(ExceptHandler)           \
	__AST_NODE_TYPE(Expression)              \
	__AST_NODE_TYPE(For)                     \
	__AST_NODE_TYPE(FormattedValue)          \
	__AST_NODE_TYPE(FunctionDefinition)      \
	__AST_NODE_TYPE(GeneratorExp)            \
	__AST_NODE_TYPE(Global)                  \
	__AST_NODE_TYPE(If)                      \
	__AST_NODE_TYPE(IfExpr)                  \
	__AST_NODE_TYPE(Import)                  \
	__AST_NODE_TYPE(ImportFrom)              \
	__AST_NODE_TYPE(JoinedStr)               \
	__AST_NODE_TYPE(Keyword)                 \
	__AST_NODE_TYPE(Lambda)                  \
	__AST_NODE_TYPE(List)                    \
	__AST_NODE_TYPE(ListComp)                \
	__AST_NODE_TYPE(Module)                  \
	__AST_NODE_TYPE(NamedExpr)               \
	__AST_NODE_TYPE(Name)                    \
	__AST_NODE_TYPE(NonLocal)                \
	__AST_NODE_TYPE(Pass)                    \
	__AST_NODE_TYPE(Raise)                   \
	__AST_NODE_TYPE(Return)                  \
	__AST_NODE_TYPE(Set)                     \
	__AST_NODE_TYPE(SetComp)                 \
	__AST_NODE_TYPE(Starred)                 \
	__AST_NODE_TYPE(Subscript)               \
	__AST_NODE_TYPE(Try)                     \
	__AST_NODE_TYPE(Tuple)                   \
	__AST_NODE_TYPE(UnaryExpr)               \
	__AST_NODE_TYPE(While)                   \
	__AST_NODE_TYPE(With)                    \
	__AST_NODE_TYPE(WithItem)                \
	__AST_NODE_TYPE(Yield)                   \
	__AST_NODE_TYPE(YieldFrom)
