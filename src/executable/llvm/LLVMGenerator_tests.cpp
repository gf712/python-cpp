#include "LLVMGenerator.hpp"
#include "LLVMProgram.hpp"

#include "lexer/Lexer.hpp"
#include "parser/Parser.hpp"

#include "gtest/gtest.h"

namespace {
std::shared_ptr<Program> generate_llvm_module(std::string_view program)
{
	spdlog::set_level(spdlog::level::info);
	auto lexer = Lexer::create(std::string(program), "_llvm_backend_tests_.py");
	parser::Parser p{ lexer };
	p.parse();

	auto module = as<ast::Module>(p.module());
	ASSERT(module)
	spdlog::set_level(spdlog::level::debug);
	return codegen::LLVMGenerator::compile(module, {}, compiler::OptimizationLevel::None);
}
}// namespace

TEST(LLVMBackend, CompilesTypeAnnotatedFunction)
{
	constexpr std::string_view program =
		"def add(a: int, b: int) -> int:\n"
		"  result = a + b\n"
		"  return result\n";

	auto llvm_module = generate_llvm_module(program);
	ASSERT_TRUE(llvm_module);
}

// TEST(LLVMBackend, CompilesTypeAnnotatedFunction)
// {
// 	constexpr std::string_view program =
// 		"def add(a: int, b: int) -> int:\n"
// 		"  result = a + b\n"
// 		"  return result\n"
// 		"print(add(1, 2))\n";

// 	auto llvm_module = generate_llvm_module(program);
// 	ASSERT_TRUE(llvm_module);

// 	const auto &func =
// 		std::static_pointer_cast<codegen::LLVMFunction>(llvm_module->as_pyfunction("add"));
// 	const auto &llvm_func = func->impl();
// }

// TEST(LLVMBackend, CompilesTypeAnnotatedFibonacci)
// {
// 	constexpr std::string_view program =
// 		"def fibo(n: int) -> int:\n"
// 		"  if n == 0:\n"
// 		"    return 0\n"
// 		"  elif n == 1:\n"
// 		"    return 1\n"
// 		"  return fibo(n-1) + fibo(n-2)\n";

// 	auto llvm_module = generate_llvm_module(program);
// 	ASSERT_TRUE(llvm_module);
// }