#pragma once

#include <variant>
#include <cstdint>

namespace ast {
class ASTNode;
}

class Bytecode;
class BytecodeGenerator;
class Interpreter;
class Instruction;
class PyObject;
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

class PyDict;
class PyFunction;
class PyModule;
class PyObject;
class PyTuple;


using Register = uint8_t;
