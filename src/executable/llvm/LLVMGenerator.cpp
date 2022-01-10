#include "LLVMGenerator.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

struct LLVMGenerator::Context
{
	LLVMContext ctx;
	IRBuilder<> builder;
	std::unique_ptr<Module> module;
};

LLVMFunction::LLVMFunction(std::unique_ptr<Module> &&module)
	: Function(0, module->getName().str(), FunctionExecutionBackend::LLVM), m_module(std::move(module))
{}

std::string LLVMFunction::to_string() const {
	std::string repr;
	raw_string_ostream out{repr};
	m_module->print(out, nullptr);
	return out.str();
}