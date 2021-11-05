#include "bytecode/BytecodeGenerator.hpp"
#include "interpreter/Interpreter.hpp"
#include "parser/Parser.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyList.hpp"
#include "runtime/PyNumber.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyString.hpp"
#include "runtime/PyTuple.hpp"
#include "vm/VM.hpp"

#include "gtest/gtest.h"

void run(std::string_view program)
{
	auto &vm = VirtualMachine::the();

	Lexer lexer{ std::string(program) };
	parser::Parser p{ lexer };
	p.parse();
	// spdlog::set_level(spdlog::level::debug);
	p.module()->print_node("");
	// spdlog::set_level(spdlog::level::info);
	auto bytecode = BytecodeGenerator::compile(p.module(), compiler::OptimizationLevel::None);
	ASSERT_FALSE(bytecode->instructions().empty());
	vm.clear();
	spdlog::info("Generated bytecode: \n{}", bytecode->to_string());
	vm.create(std::move(bytecode));
	// spdlog::set_level(spdlog::level::debug);
	vm.execute();
	// spdlog::set_level(spdlog::level::debug);
	// vm.dump();
	// spdlog::set_level(spdlog::level::info);
}


template<typename T> struct is_vector : std::false_type
{
};

template<typename T> struct is_vector<std::vector<T>> : std::true_type
{
};

template<typename T> struct is_unordered_map : std::false_type
{
};

template<typename T, typename U> struct is_unordered_map<std::unordered_map<T, U>> : std::true_type
{
};

template<typename T> void check_value(const PyObject *obj, T expected_value)
{
	if constexpr (std::is_integral_v<T>) {
		ASSERT_EQ(obj->type(), PyObjectType::PY_NUMBER);
		auto pynum = as<PyNumber>(obj);
		ASSERT_TRUE(pynum);
		ASSERT_EQ(std::get<int64_t>(pynum->value().value), expected_value);
	} else if constexpr (std::is_same_v<T, const char *> || std::is_same_v<T, std::string>) {
		ASSERT_EQ(obj->type(), PyObjectType::PY_STRING);
		auto pystring = as<PyString>(obj);
		ASSERT_TRUE(pystring);
		ASSERT_EQ(pystring->value(), expected_value);
	} else {
		TODO()
	}
}

template<typename T> void assert_interpreter_object_value(std::string name, T expected_value)
{
	auto &vm = VirtualMachine::the();
	for (const auto &[k, v] : vm.interpreter()->execution_frame()->locals()->map()) {
		spdlog::debug(
			"Key: {}, Value: {}", PyObject::from(k)->to_string(), PyObject::from(v)->to_string());
	}
	auto value = vm.interpreter()->execution_frame()->locals()->map().at(String{ name });
	ASSERT_TRUE(std::holds_alternative<PyObject *>(value));
	auto obj = std::get<PyObject *>(value);
	ASSERT_TRUE(obj);
	if constexpr (is_vector<T>{}) {
		ASSERT_EQ(obj->type(), PyObjectType::PY_LIST);
		auto pylist = as<PyList>(obj);
		ASSERT_TRUE(pylist);
		size_t i = 0;
		for (const auto &el : pylist->elements()) {
			std::visit(
				overloaded{ [&](const PyObject *obj) { check_value(obj, expected_value[i]); },
					[&](const auto &value) {
						check_value(PyObject::from(value), expected_value[i]);
					} },
				el);
			i++;
		}
	} else if constexpr (is_unordered_map<T>{}) {
		ASSERT_EQ(obj->type(), PyObjectType::PY_DICT);
		auto pydict = as<PyDict>(obj);
		ASSERT_TRUE(pydict);
		// FIXME: this prolongs the lifetime of items should be just be:
		//		  for (const auto &p : pydict->items()) {...}
		auto items = pydict->items();
		for (const auto &p : *items) {
			auto key = p->operator[](0);
			auto value = p->operator[](1);
			// only support string keys for now
			ASSERT(as<PyString>(key))
			auto key_string = as<PyString>(key)->value();
			check_value(value, expected_value[key_string]);
		}
	} else {
		check_value(obj, expected_value);
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
	assert_interpreter_object_value("a", 2 * 3 + 4 * 5 * 6);
}

TEST(RunPythonProgram, AssignmentWithBitshift)
{
	constexpr std::string_view program = "a = 1 + 2 + 5 * 2 << 10 * 2 + 1\n";
	run(program);
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

TEST(RunPythonProgram, IfBlockAssignmentInBody)
{
	constexpr std::string_view program =
		"if True:\n"
		"	a = 1\n"
		"else:\n"
		"	a = 2\n";

	run(program);
	assert_interpreter_object_value("a", 1);
}

TEST(RunPythonProgram, IfBlockAssignmentInOrElse)
{
	constexpr std::string_view program =
		"if None:\n"
		"	a = 1\n"
		"else:\n"
		"	a = 2\n";

	run(program);
	assert_interpreter_object_value("a", 2);
}


TEST(RunPythonProgram, FunctionWithIfElseAssignment)
{
	static constexpr std::string_view program =
		"def foo(a):\n"
		"	if a == 1:\n"
		"		result = 10\n"
		"	else:\n"
		"		result = 2\n"
		"	return result\n"
		"a = foo(1)\n"
		"b = foo(5)\n";

	run(program);
	assert_interpreter_object_value("a", 10);
	assert_interpreter_object_value("b", 2);
}


TEST(RunPythonProgram, FunctionWithIfElseReturn)
{
	static constexpr std::string_view program =
		"def foo(a):\n"
		"	if a == 1:\n"
		"		return 10\n"
		"	else:\n"
		"		return 2\n"
		"a = foo(1)\n"
		"b = foo(5)\n";

	run(program);
	assert_interpreter_object_value("a", 10);
	assert_interpreter_object_value("b", 2);
}

TEST(RunPythonProgram, IfElifElseInModuleSpace)
{
	static constexpr std::string_view program1 =
		"a = 0\n"
		"if a == 1:\n"
		"	a = 0\n"
		"elif a == 2:\n"
		"	a = 2\n"
		"else:\n"
		"	a = 3\n";
	run(program1);
	assert_interpreter_object_value("a", 3);

	static constexpr std::string_view program2 =
		"a = 1\n"
		"if a == 1:\n"
		"	a = 0\n"
		"elif a == 2:\n"
		"	a = 2\n"
		"else:\n"
		"	a = 3\n";
	run(program2);
	assert_interpreter_object_value("a", 0);

	static constexpr std::string_view program3 =
		"a = 2\n"
		"if a == 1:\n"
		"	a = 0\n"
		"elif a == 2:\n"
		"	a = 5\n"
		"else:\n"
		"	a = 3\n";
	run(program3);
	assert_interpreter_object_value("a", 5);
}


TEST(RunPythonProgram, BuildListLiteralWithValues)
{
	static constexpr std::string_view program = "a = [1,2,3,5]\n";

	run(program);
	assert_interpreter_object_value("a", std::vector<int64_t>{ 1, 2, 3, 5 });
}


TEST(RunPythonProgram, BuildDictLiteralWithValues)
{
	static constexpr std::string_view program = "a = {\"a\": 1}\n";

	run(program);
	assert_interpreter_object_value(
		"a", std::unordered_map<std::string, int64_t>{ { "a", int64_t{ 1 } } });
}


TEST(RunPythonProgram, FibonacciRecursive)
{
	// FIXME: currently requires assignment to local lhs/rhs
	//        instead of operating directly on the function call
	static constexpr std::string_view program =
		"def fibonacci(element):\n"
		"	if element == 0:\n"
		"		return 0\n"
		"	if element == 1:\n"
		"		return 1\n"
		"	else:\n"
		"		lhs = fibonacci(element-1)\n"
		"		rhs = fibonacci(element-2)\n"
		"		return lhs + rhs\n"
		"s1 = fibonacci(1)\n"
		"s2 = fibonacci(2)\n"
		"s3 = fibonacci(3)\n"
		"s4 = fibonacci(4)\n"
		"s5 = fibonacci(5)\n"
		"s6 = fibonacci(6)\n"
		"s7 = fibonacci(7)\n"
		"s8 = fibonacci(8)\n"
		"s9 = fibonacci(9)\n";

	run(program);
	assert_interpreter_object_value("s1", 1);
	assert_interpreter_object_value("s2", 1);
	assert_interpreter_object_value("s3", 2);
	assert_interpreter_object_value("s4", 3);
	assert_interpreter_object_value("s5", 5);
	assert_interpreter_object_value("s6", 8);
	assert_interpreter_object_value("s7", 13);
	assert_interpreter_object_value("s8", 21);
	assert_interpreter_object_value("s9", 34);
}

TEST(RunPythonProgram, ForLoopWithAccumulator)
{
	static constexpr std::string_view program =
		"acc = 0\n"
		"for x in [1,2,3,4]:\n"
		"	acc = acc + x\n";

	run(program);
	assert_interpreter_object_value("acc", 10);
}

TEST(RunPythonProgram, ForLoopWithRange)
{
	static constexpr std::string_view program =
		"acc = 0\n"
		"for x in range(100):\n"
		"	acc = acc + x\n";

	run(program);
	assert_interpreter_object_value("acc", 4950);
}


TEST(RunPythonProgram, ForLoopAccumulateEvenAndOddNumbers)
{
	static constexpr std::string_view program =
		"acc_even = 0\n"
		"acc_odd = 0\n"
		"for x in range(100):\n"
		"	if x % 2 == 0:\n"
		"		acc_even = acc_even + x\n"
		"	else:\n"
		"		acc_odd = acc_odd + x\n";

	run(program);
	assert_interpreter_object_value("acc_even", 2450);
	assert_interpreter_object_value("acc_odd", 2500);
}


TEST(RunPythonProgram, AccessClassAttribute)
{
	static constexpr std::string_view program =
		"class A:\n"
		"	a = 1\n"
		"foo = A()\n"
		"result = foo.a\n";

	run(program);
	assert_interpreter_object_value("result", 1);
}


TEST(RunPythonProgram, CallClassMethod)
{
	static constexpr std::string_view program =
		"class A:\n"
		"	a = 1\n"
		"	def plus_a(self, a):\n"
		"		return self.a + a\n"
		"foo = A()\n"
		"result = foo.plus_a(41)\n";

	run(program);
	assert_interpreter_object_value("result", 42);
}


TEST(RunPythonProgram, UpdateAttributeValue)
{
	static constexpr std::string_view program =
		"class A:\n"
		"	a = 1\n"
		"	def plus_a(self, a):\n"
		"		return self.a + a\n"
		"foo = A()\n"
		"foo.a = 20\n"
		"result = foo.plus_a(22)\n";

	run(program);
	assert_interpreter_object_value("result", 42);
}


TEST(RunPythonProgram, CallFunctionWithKeyword)
{
	static constexpr std::string_view program =
		"def sub(lhs, rhs):\n"
		"   return lhs - rhs\n"
		"c = sub(rhs=3, lhs=10)\n"
		"d = sub(lhs=3, rhs=10)\n"
		"e = sub(0, rhs=19)\n";

	run(program);
	assert_interpreter_object_value("c", 7);
	assert_interpreter_object_value("d", -7);
	assert_interpreter_object_value("e", -19);
}


TEST(RunPythonProgram, AugmentedAssign)
{
	static constexpr std::string_view program =
		"a = 1\n"
		"b = 10\n"
		"a += b\n";

	run(program);
	assert_interpreter_object_value("a", 11);
	assert_interpreter_object_value("b", 10);
}


TEST(RunPythonProgram, BuiltinOrd)
{
	static constexpr std::string_view program = "smiley_codepoint = ord(\"😃\")\n";

	run(program);
	assert_interpreter_object_value("smiley_codepoint", 128515);
}

TEST(RunPythonProgram, WhileLoop)
{
	static constexpr std::string_view program =
		"acc = 0\n"
		"i = 0\n"
		"while i <= 10:\n"
		"	acc += i\n"
		"	i += 1\n";

	run(program);
	assert_interpreter_object_value("acc", 55);
}


TEST(RunPythonProgram, UnpackAssignment)
{
	static constexpr std::string_view program = "a, b = 1, 2\n";

	run(program);
	assert_interpreter_object_value("a", 1);
	assert_interpreter_object_value("b", 2);
}
