#include "bytecode/BytecodeGenerator.hpp"
#include "bytecode/VM.hpp"
#include "interpreter/Interpreter.hpp"
#include "parser/Parser.hpp"
#include "runtime/PyObject.hpp"

#include "gtest/gtest.h"

void run(std::string_view program)
{
	Lexer lexer{ std::string(program) };
	parser::Parser p{ lexer };
	p.parse();
	// spdlog::set_level(spdlog::level::debug);
	// p.module()->print_node("");
	// spdlog::set_level(spdlog::level::critical);
	auto bytecode = BytecodeGenerator::compile(p.module());
	ASSERT_FALSE(bytecode->instructions().empty());
	auto &vm = VirtualMachine::the();
	vm.clear();
	std::cout << bytecode->to_string() << '\n';
	vm.create(std::move(bytecode));
	vm.execute();
	spdlog::set_level(spdlog::level::debug);
	vm.dump();
	spdlog::set_level(spdlog::level::critical);
}


template<typename T> void assert_interpreter_object_value(std::string name, T expected_value)
{
	auto &vm = VirtualMachine::the();
	auto obj = vm.interpreter()->fetch_object(name);
	ASSERT_TRUE(obj);
	if constexpr (std::is_integral_v<T>) {
		ASSERT_EQ(obj->type(), PyObjectType::PY_NUMBER);
		auto pynum = as<PyObjectNumber>(obj);
		ASSERT_TRUE(pynum);
		ASSERT_EQ(std::get<int64_t>(pynum->value().value), expected_value);
	} else if constexpr (std::is_same_v<T, const char *> || std::is_same_v<T, std::string>) {
		ASSERT_EQ(obj->type(), PyObjectType::PY_STRING);
		auto pystring = as<PyString>(obj);
		ASSERT_TRUE(pystring);
		ASSERT_EQ(pystring->value(), expected_value);
	} else {
		ASSERT_TRUE(false);
	}
}


TEST(RunPythonProgram, SimpleAssignment)
{
	constexpr std::string_view program = "a = 2\n";
	run(program);
	auto &vm = VirtualMachine::the();
	ASSERT_EQ(std::get<int64_t>(std::get<Number>(vm.reg(1)).value), 2);

	assert_interpreter_object_value("a", 2);
}

TEST(RunPythonProgram, SimplePowerAssignment)
{
	constexpr std::string_view program = "a = 2 ** 10\n";
	run(program);
	auto &vm = VirtualMachine::the();
	ASSERT_EQ(std::get<int64_t>(std::get<Number>(vm.reg(1)).value), 2);
	ASSERT_EQ(std::get<int64_t>(std::get<Number>(vm.reg(2)).value), 10);
	ASSERT_EQ(std::get<int64_t>(std::get<Number>(vm.reg(3)).value), std::pow(2, 10));

	assert_interpreter_object_value("a", static_cast<int64_t>(std::pow(2, 10)));
}

TEST(RunPythonProgram, AssignmentWithMultiplicationPrecedence)
{
	constexpr std::string_view program = "a = 2 * 3 + 4 * 5 * 6\n";
	run(program);
	auto &vm = VirtualMachine::the();

	assert_interpreter_object_value("a", 2 * 3 + 4 * 5 * 6);
}

TEST(RunPythonProgram, AssignmentWithBitshift)
{
	constexpr std::string_view program = "a = 1 + 2 + 5 * 2 << 10 * 2 + 1\n";
	run(program);
	auto &vm = VirtualMachine::the();

	assert_interpreter_object_value("a", (1 + 2 + 5 * 2) << (10 * 2 + 1));
}

TEST(RunPythonProgram, MultilineNumericAssignments)
{
	constexpr std::string_view program =
		"a = 15 + 22 - 1\n"
		"b = a\n"
		"c = a + b\n";
	run(program);

	assert_interpreter_object_value("a", 36);
	assert_interpreter_object_value("b", 36);
	assert_interpreter_object_value("c", 72);
}

TEST(RunPythonProgram, MultilineStringAssignments)
{
	constexpr std::string_view program =
		"a = \"foo\"\n"
		"b = \"bar\"\n"
		"c = \"123\"\n"
		"d = a + b + c\n";
	run(program);

	assert_interpreter_object_value("a", "foo");
	assert_interpreter_object_value("b", "bar");
	assert_interpreter_object_value("c", "123");
	assert_interpreter_object_value("d", "foobar123");
}


TEST(RunPythonProgram, AddFunctionDeclarationAndCall)
{
	constexpr std::string_view program =
		"def add(a, b):\n"
		"	return a + b\n"
		"a = 3\n"
		"b = 10\n"
		"c = add(a, b)\n"
		"a = 5\n"
		"d = add(a, b)\n"
		"e = add(a, d)\n";
	run(program);

	assert_interpreter_object_value("a", 5);
	assert_interpreter_object_value("b", 10);
	assert_interpreter_object_value("c", 13);
	assert_interpreter_object_value("d", 15);
	assert_interpreter_object_value("e", 20);
}


TEST(RunPythonProgram, BuiltinPrintFunction)
{
	constexpr std::string_view program =
		"a = 20\n"
		"b = 22\n"
		"c = a + b\n"
		"r = print(a)\n"
		"print(r)\n"
		"print(b)\n"
		"print(c)\n";
	run(program);
	assert_interpreter_object_value("a", 20);
	assert_interpreter_object_value("b", 22);
	assert_interpreter_object_value("c", 42);
}


TEST(RunPythonProgram, AddFunction)
{
	constexpr std::string_view program =
		"def add(lhs, rhs):\n"
		"   return lhs + rhs\n"
		"a = 1\n"
		"b = 2\n"
		"c_with_variables = add(a, b)\n"
		"c_with_constants = add(3, 19)\n"
		"c_mixed = add(a, 19)\n"
		"c_rvalue = add(a+1, 39+1)\n";
	run(program);
	assert_interpreter_object_value("a", 1);
	assert_interpreter_object_value("b", 2);
	assert_interpreter_object_value("c_with_variables", 3);
	assert_interpreter_object_value("c_with_constants", 22);
	assert_interpreter_object_value("c_rvalue", 42);
}

TEST(RunPythonProgram, MultilineFunction)
{
	constexpr std::string_view program =
		"def plus_one(value):\n"
		"   constant = 1\n"
		"   return value + constant\n"
		"c = plus_one(10)\n";
	run(program);
	assert_interpreter_object_value("c", 11);
}


TEST(RunPythonProgram, FunctionArgAssignment)
{
	constexpr std::string_view program =
		"def plus_one(value):\n"
		"   value = value + 1\n"
		"   return value\n"
		"c = plus_one(10)\n";
	run(program);
	assert_interpreter_object_value("c", 11);
}
