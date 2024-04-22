#pragma once

#include <cstdint>
#include <variant>

namespace ast {
class ASTNode;
class Module;
}// namespace ast

class Bytecode;
class BytecodeProgram;
namespace codegen {
class BytecodeGenerator;
}
class Function;
class Interpreter;
class InterpreterSession;
class Instruction;
class Program;
class VirtualMachine;
class Label;

namespace parser {
class Parser;
}

struct Load;

namespace py {
struct Number;
struct String;
struct Bytes;
struct Ellipsis;
struct NoneType;
struct NameConstant;
struct Tuple;
class PyObject;
using Value = std::variant<Number, String, Bytes, Ellipsis, NameConstant, Tuple, PyObject *>;
template<typename T> class PyResult;
}// namespace py

using Register = uint8_t;
