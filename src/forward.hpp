#pragma once

#include <cstdint>
#include <variant>

namespace ast {
class ASTNode;
class Module;
}// namespace ast

class Bytecode;
namespace codegen {
class BytecodeGenerator;
}
class Function;
class Interpreter;
class InterpreterSession;
class Instruction;
class Program;

namespace parser {
class Parser;
}

struct Load;

struct Number;
struct String;
struct Bytes;
struct Ellipsis;
struct NoneType;
struct NameConstant;

class PyObject;
using Value = std::variant<Number, String, Bytes, Ellipsis, NameConstant, PyObject *>;

using Register = uint8_t;
