# Python C++

A Python interpreter implementation in C++. The current aim is to be compliant with the Python 3.10 spec and have releases inline with future Python versions.

# What is different from CPython?

 * This interpreter uses a register-based VM, instead of stack-based VM
 * JIT support (TODO: the plan is to use something like LLVM or ASMJIT to support some code optimisations at runtime)
 * The runtime is written in C++, so Python Objects are C++ classes internally. This should make it easier to write C++ Python bindings.
 * No manual reference counting (use RAAI instead)

# What doesn't change from CPython?

 * Tokens generated by the Lexer are the same as in CPython (see [token list](https://docs.python.org/3/library/token.html))
 * The grammar specification is the same as the one used to generate the CPython parser (see [spec](https://docs.python.org/3/reference/grammar.html))
 * The AST nodes are the same ones used in CPython (see [node definitions](https://greentreesnakes.readthedocs.io/en/latest/nodes.html#))
