#include "LLVMProgram.hpp"
#include "executable/Function.hpp"
#include "executable/llvm/LLVMGenerator.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

LLVMProgram::LLVMProgram(std::unique_ptr<llvm::Module> &&module,
	std::unique_ptr<llvm::LLVMContext> &&ctx,
	std::string filename,
	std::vector<std::string> argv)
	: Program(std::move(filename), std::move(argv)), m_ctx(std::move(ctx)),
	  m_module(std::move(module))
{
	for (const auto &f : m_module->functions()) {
		m_functions.push_back(std::make_shared<codegen::LLVMFunction>(f));
	}
}

std::string LLVMProgram::to_string() const
{
	std::string repr;
	raw_string_ostream out{ repr };
	m_module->print(out, nullptr);
	return out.str();
}

int LLVMProgram::execute(VirtualMachine *) { TODO(); }

const std::shared_ptr<::Function> &LLVMProgram::function(const std::string &name) const
{
	for (const auto &f : m_functions) {
		if (f->function_name() == name) { return f; }
	}
	ASSERT(false)
}