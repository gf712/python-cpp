#include "LLVMProgram.hpp"
#include "LLVMPyUtils.hpp"
#include "executable/Function.hpp"
#include "executable/llvm/LLVMGenerator.hpp"
#include "runtime/Value.hpp"

#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

namespace py {
class PyTuple;
class PyDict;
}// namespace py

using namespace llvm;
using namespace py;


struct LLVMProgram::InteropFunctions
{
	// from Python to fundamental types
	llvm::Function *F_object_to_i64;

	// from fundamental types to Python
	llvm::Function *F_i64_to_object;

	static std::unique_ptr<InteropFunctions> create(Module *module,
		LLVMContext &ctx,
		orc::JITDylib &lib,
		orc::ExecutionSession &es,
		const DataLayout &dl)
	{
		auto interop = std::make_unique<InteropFunctions>();

		auto *pyobject_type = Type::getInt8PtrTy(ctx);
		orc::MangleAndInterner mangle(es, dl);

		// int64_t from_args(PyTuple*, size_t index);
		{
			auto *return_type = Type::getInt64Ty(ctx);
			std::vector<Type *> arg_types{ pyobject_type, Type::getInt64Ty(ctx) };

			auto *FT = FunctionType::get(return_type, arg_types, false);
			interop->F_object_to_i64 = llvm::Function::Create(
				FT, llvm::Function::ExternalLinkage, "from_args_i64", module);

			cantFail(lib.define(orc::absoluteSymbols(orc::SymbolMap{ { mangle("from_args_i64"),
				JITEvaluatedSymbol{
					pointerToJITTargetAddress(&from_args_i64), JITSymbolFlags::Exported } } })));
		}

		// PyObject *from(int64_t value);
		{
			auto *return_type = pyobject_type;
			std::vector<Type *> arg_types{ Type::getInt64Ty(ctx) };

			auto *FT = FunctionType::get(return_type, arg_types, false);
			interop->F_i64_to_object =
				llvm::Function::Create(FT, llvm::Function::ExternalLinkage, "from_i64", module);

			cantFail(lib.define(orc::absoluteSymbols(orc::SymbolMap{ { mangle("from_i64"),
				JITEvaluatedSymbol{
					pointerToJITTargetAddress(&from_i64), JITSymbolFlags::Exported } } })));
		}

		return interop;
	}
};

struct LLVMProgram::InternalConfig
{
	std::unique_ptr<orc::ExecutionSession> execution_session;
	std::unique_ptr<orc::RTDyldObjectLinkingLayer> object_layer;
	orc::JITDylib &main_jitlib;
	std::unique_ptr<orc::IRCompileLayer> compile_layer;
	std::unique_ptr<DataLayout> data_layout;

	InternalConfig(std::unique_ptr<orc::ExecutionSession> &&es,
		std::unique_ptr<orc::RTDyldObjectLinkingLayer> &&ol,
		std::unique_ptr<orc::IRCompileLayer> &&cl,
		orc::JITDylib &jitlib,
		std::unique_ptr<DataLayout> &&dl)
		: execution_session(std::move(es)), object_layer(std::move(ol)), main_jitlib(jitlib),
		  compile_layer(std::move(cl)), data_layout(std::move(dl)){};

	static std::unique_ptr<InternalConfig> create(std::unique_ptr<llvm::Module> &&module,
		std::unique_ptr<llvm::LLVMContext> &&ctx)
	{
		InitializeNativeTarget();
		InitializeNativeTargetAsmPrinter();
		InitializeNativeTargetAsmParser();

		auto target = orc::JITTargetMachineBuilder::detectHost();
		if (!target) {
			(void)handleErrors(
				target.takeError(), [](const ErrorInfoBase &e) { spdlog::error(e.message()); });
		}
		auto execution_session = std::make_unique<orc::ExecutionSession>();

		auto object_layer = std::make_unique<orc::RTDyldObjectLinkingLayer>(
			*execution_session, []() { return std::make_unique<SectionMemoryManager>(); });

		std::unique_ptr<orc::IRCompileLayer::IRCompiler> compiler =
			std::make_unique<orc::ConcurrentIRCompiler>(*target);
		auto compile_layer = std::make_unique<orc::IRCompileLayer>(
			*execution_session, *object_layer, std::move(compiler));
		auto &main_jitlib = execution_session->createJITDylib("main");
		cantFail(compile_layer->add(
			main_jitlib, orc::ThreadSafeModule(std::move(module), std::move(ctx))));

		auto target_default_data_layout = target->getDefaultDataLayoutForTarget();
		if (!target_default_data_layout) {
			(void)handleErrors(target_default_data_layout.takeError(),
				[](const ErrorInfoBase &e) { spdlog::error(e.message()); });
			TODO();
		}

		auto data_layout = std::make_unique<DataLayout>(*target_default_data_layout);

		return std::make_unique<InternalConfig>(std::move(execution_session),
			std::move(object_layer),
			std::move(compile_layer),
			main_jitlib,
			std::move(data_layout));
	}
};

LLVMProgram::LLVMProgram(std::unique_ptr<llvm::Module> &&module,
	std::unique_ptr<llvm::LLVMContext> &&ctx,
	std::string filename,
	std::vector<std::string> argv)
	: Program(std::move(filename), std::move(argv))
{
	for (const auto &f : module->functions()) {
		m_functions.push_back(std::make_shared<codegen::LLVMFunction>(f));
	}
	auto *m = module.get();
	auto *ctx_ptr = ctx.get();

	m_config = LLVMProgram::InternalConfig::create(std::move(module), std::move(ctx)).release();
	m_interop_functions = LLVMProgram::InteropFunctions::create(
		m, *ctx_ptr, m_config->main_jitlib, *m_config->execution_session, *m_config->data_layout);
}

LLVMProgram::~LLVMProgram() { delete m_config; }

std::string LLVMProgram::to_string() const
{
	std::string repr;
	raw_string_ostream out{ repr };
	// m_config->main_jitlib-> ->print(out, nullptr);
	m_config->main_jitlib.dump(out);
	return out.str();
}

int LLVMProgram::execute(VirtualMachine *) { TODO(); }

void LLVMProgram::create_interop_function(const std::shared_ptr<::Function> &func,
	const std::string &mangled_name) const
{
	auto ctx = std::make_unique<LLVMContext>();
	IRBuilder<> builder{ *ctx };
	auto module = std::make_unique<Module>("glue_{}" + mangled_name, *ctx);

	// pointer to PyObject*
	auto *pyobject_type = Type::getInt8PtrTy(*ctx);

	auto *return_type = pyobject_type;
	std::vector<Type *> arg_types{ pyobject_type, pyobject_type };

	// function as a proxy to function signature PyObject*(PyTuple*, PyDict*)
	// this is actually int8*(int8*, int8*)
	auto *FT = FunctionType::get(return_type, arg_types, false);
	auto *F = llvm::Function::Create(
		FT, llvm::Function::ExternalLinkage, "glue_" + mangled_name, module.get());
	auto *BB = BasicBlock::Create(*ctx, "entry", F);
	builder.SetInsertPoint(BB);

	std::vector<llvm::Value *> args;
	const auto &func_impl = static_cast<const codegen::LLVMFunction *>(func.get())->impl();
	size_t idx{ 0 };
	for (const auto &arg : func_impl.args()) {
		auto *type = arg.getType();
		if (type->isIntegerTy(64)) {
			// get element at index idx from PyTuple* args as a int64_t
			auto *args_tuple = F->getArg(0);
			auto *tuple_idx = ConstantInt::get(Type::getInt64Ty(*ctx), idx);
			args.push_back(builder.CreateCall(m_interop_functions->F_object_to_i64,
				{ args_tuple, tuple_idx },
				"from_args_value"));
		} else {
			type->print(outs());
			std::cout << '\n';
			TODO();
		}
		idx++;
	}

	auto *result =
		builder.CreateCall(const_cast<llvm::Function *>(&func_impl), args, mangled_name + "result");

	auto *result_obj = [&, this]() -> llvm::Value * {
		// check if we can deal with return type
		if (result->getType()->isIntegerTy(64)) {
			return builder.CreateCall(
				m_interop_functions->F_i64_to_object, { result }, "result_obj");
		} else {
			TODO();
		}
		return nullptr;
	}();

	auto *return_value = builder.CreateRet(result_obj);
	(void)return_value;

	cantFail(m_config->compile_layer->add(*m_config->execution_session->getJITDylibByName("main"),
		orc::ThreadSafeModule(std::move(module), std::move(ctx))));
}

py::PyObject *LLVMProgram::as_pyfunction(const std::string &function_name,
	const std::vector<std::string> &argnames,
	const std::vector<py::Value> &default_values,
	const std::vector<py::Value> &kw_default_values,
	size_t positional_args_count,
	size_t kwonly_args_count,
	const CodeFlags &flags) const
{
	if (!default_values.empty()) { TODO(); }
	if (!kw_default_values.empty()) { TODO(); }

	(void)argnames;
	(void)positional_args_count;
	(void)kwonly_args_count;
	(void)flags;

	const auto start = function_name.find_last_of('.') + 1;
	const std::string demangled_name{ function_name.begin() + start, function_name.end() };
	const auto &function = [this, &demangled_name]() -> const std::shared_ptr<::Function> {
		for (const auto &f : m_functions) {
			if (f->function_name() == demangled_name) { return f; }
		}
		return nullptr;
	}();

	if (!function) { return nullptr; }

	create_interop_function(function, function_name);

	auto maybe_symbol =
		m_config->execution_session->lookup({ &m_config->main_jitlib }, "glue_" + function_name);
	if (!maybe_symbol) {
		(void)handleErrors(
			maybe_symbol.takeError(), [](const ErrorInfoBase &e) { spdlog::error(e.message()); });
		TODO();
	}

	std::function<PyObject *(PyTuple *, PyDict *)> llvm_func =
		jitTargetAddressToFunction<PyObject *(*)(PyTuple *, PyDict *)>(maybe_symbol->getAddress());

	return create_native_function(demangled_name, std::move(llvm_func));
}