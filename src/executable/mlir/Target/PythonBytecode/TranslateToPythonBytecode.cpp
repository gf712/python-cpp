#include "Dialect/EmitPythonBytecode/IR/EmitPythonBytecode.hpp"
#include "Dialect/Python/IR/PythonOps.hpp"
#include "Target/PythonBytecode/PythonBytecodeEmitter.hpp"
#include "executable/Mangler.hpp"
#include "executable/bytecode/BytecodeProgram.hpp"
#include "executable/bytecode/instructions/BinaryOperation.hpp"
#include "executable/bytecode/instructions/BinarySubscript.hpp"
#include "executable/bytecode/instructions/BuildDict.hpp"
#include "executable/bytecode/instructions/BuildList.hpp"
#include "executable/bytecode/instructions/BuildSet.hpp"
#include "executable/bytecode/instructions/BuildSlice.hpp"
#include "executable/bytecode/instructions/BuildString.hpp"
#include "executable/bytecode/instructions/BuildTuple.hpp"
#include "executable/bytecode/instructions/ClearExceptionState.hpp"
#include "executable/bytecode/instructions/CompareOperation.hpp"
#include "executable/bytecode/instructions/DeleteAttr.hpp"
#include "executable/bytecode/instructions/DeleteDeref.hpp"
#include "executable/bytecode/instructions/DeleteFast.hpp"
#include "executable/bytecode/instructions/DeleteGlobal.hpp"
#include "executable/bytecode/instructions/DeleteName.hpp"
#include "executable/bytecode/instructions/DeleteSubscript.hpp"
#include "executable/bytecode/instructions/DictAdd.hpp"
#include "executable/bytecode/instructions/DictUpdate.hpp"
#include "executable/bytecode/instructions/ForIter.hpp"
#include "executable/bytecode/instructions/FormatValue.hpp"
#include "executable/bytecode/instructions/FunctionCall.hpp"
#include "executable/bytecode/instructions/FunctionCallEx.hpp"
#include "executable/bytecode/instructions/FunctionCallWithKeywords.hpp"
#include "executable/bytecode/instructions/GetAwaitable.hpp"
#include "executable/bytecode/instructions/GetIter.hpp"
#include "executable/bytecode/instructions/GetYieldFromIter.hpp"
#include "executable/bytecode/instructions/ImportFrom.hpp"
#include "executable/bytecode/instructions/ImportName.hpp"
#include "executable/bytecode/instructions/ImportStar.hpp"
#include "executable/bytecode/instructions/InplaceOp.hpp"
#include "executable/bytecode/instructions/Instructions.hpp"
#include "executable/bytecode/instructions/Jump.hpp"
#include "executable/bytecode/instructions/JumpIfExceptionMatch.hpp"
#include "executable/bytecode/instructions/JumpIfFalse.hpp"
#include "executable/bytecode/instructions/JumpIfNotExceptionMatch.hpp"
#include "executable/bytecode/instructions/JumpIfTrue.hpp"
#include "executable/bytecode/instructions/LeaveExceptionHandling.hpp"
#include "executable/bytecode/instructions/ListAppend.hpp"
#include "executable/bytecode/instructions/ListExtend.hpp"
#include "executable/bytecode/instructions/ListToTuple.hpp"
#include "executable/bytecode/instructions/LoadAssertionError.hpp"
#include "executable/bytecode/instructions/LoadAttr.hpp"
#include "executable/bytecode/instructions/LoadBuildClass.hpp"
#include "executable/bytecode/instructions/LoadClosure.hpp"
#include "executable/bytecode/instructions/LoadConst.hpp"
#include "executable/bytecode/instructions/LoadDeref.hpp"
#include "executable/bytecode/instructions/LoadFast.hpp"
#include "executable/bytecode/instructions/LoadGlobal.hpp"
#include "executable/bytecode/instructions/LoadMethod.hpp"
#include "executable/bytecode/instructions/LoadName.hpp"
#include "executable/bytecode/instructions/MakeFunction.hpp"
#include "executable/bytecode/instructions/Move.hpp"
#include "executable/bytecode/instructions/Pop.hpp"
#include "executable/bytecode/instructions/Push.hpp"
#include "executable/bytecode/instructions/RaiseVarargs.hpp"
#include "executable/bytecode/instructions/ReRaise.hpp"
#include "executable/bytecode/instructions/ReturnValue.hpp"
#include "executable/bytecode/instructions/SetAdd.hpp"
#include "executable/bytecode/instructions/SetUpdate.hpp"
#include "executable/bytecode/instructions/SetupExceptionHandling.hpp"
#include "executable/bytecode/instructions/SetupWith.hpp"
#include "executable/bytecode/instructions/StoreAttr.hpp"
#include "executable/bytecode/instructions/StoreDeref.hpp"
#include "executable/bytecode/instructions/StoreFast.hpp"
#include "executable/bytecode/instructions/StoreGlobal.hpp"
#include "executable/bytecode/instructions/StoreName.hpp"
#include "executable/bytecode/instructions/StoreSubscript.hpp"
#include "executable/bytecode/instructions/ToBool.hpp"
#include "executable/bytecode/instructions/Unary.hpp"
#include "executable/bytecode/instructions/UnpackExpand.hpp"
#include "executable/bytecode/instructions/UnpackSequence.hpp"
#include "executable/bytecode/instructions/WithExceptStart.hpp"
#include "executable/bytecode/instructions/YieldFrom.hpp"
#include "executable/bytecode/instructions/YieldLoad.hpp"
#include "executable/bytecode/instructions/YieldValue.hpp"
#include "executable/mlir/Target/PythonBytecode/LinearScanRegisterAllocation.hpp"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "runtime/Value.hpp"
#include "utilities.hpp"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <map>
#include <optional>
#include <ranges>

using namespace mlir;

namespace codegen {

struct PythonBytecodeEmitter
{
	struct FunctionInfo
	{
		std::vector<std::unique_ptr<Instruction>> instructions;
		std::vector<InstructionSourceLocation> instruction_locations;
		std::vector<::py::Value> m_consts;
		std::vector<std::string> m_varnames;
		std::vector<std::string> m_names;
		std::vector<std::string> m_cellvars;
		std::vector<std::string> m_freevars;
		std::vector<size_t> m_cell2arg;
		size_t m_arg_count{ 0 };
		size_t m_kwonly_arg_count{ 0 };
		size_t m_stack_size{ 0 };
		CodeFlags m_flags = CodeFlags::create();

		void set_varargs() { m_flags.set(CodeFlags::Flag::VARARGS); }

		void set_kwargs() { m_flags.set(CodeFlags::Flag::VARKEYWORDS); }

		void set_is_generator() { m_flags.set(CodeFlags::Flag::GENERATOR); }

		void set_is_async() { m_flags.set(CodeFlags::Flag::COROUTINE); }

		void set_is_class() { m_flags.set(CodeFlags::Flag::CLASS); }

		void set_kwonlyarg_count(size_t kwonlyarg_count) { m_kwonly_arg_count = kwonlyarg_count; }

		size_t add_const(::py::Value value)
		{
			if (auto it = std::find_if(m_consts.begin(),
					m_consts.end(),
					[&value](const auto &el) {
						if (std::holds_alternative<::py::Number>(el)
							&& std::holds_alternative<::py::Number>(value)) {
							return std::get<::py::Number>(el).value.index()
									   == std::get<::py::Number>(value).value.index()
								   && el == value;
						}
						return el.index() == value.index() && el == value;
					});
				it != m_consts.end()) {
				return std::distance(m_consts.begin(), it);
			}
			m_consts.push_back(std::move(value));
			return m_consts.size() - 1;
		}

		size_t add_name(std::string_view str)
		{
			auto it = std::find_if(
				m_names.begin(), m_names.end(), [&str](const auto &el) { return el == str; });
			if (it == m_names.end()) {
				m_names.emplace_back(str);
				return m_names.size() - 1;
			}
			return std::distance(m_names.begin(), it);
		}

		size_t get_cell_index(std::string_view str)
		{
			auto it = std::find_if(
				m_cellvars.begin(), m_cellvars.end(), [&str](const auto &el) { return el == str; });
			if (it != m_cellvars.end()) { return std::distance(m_cellvars.begin(), it); }

			it = std::find_if(
				m_freevars.begin(), m_freevars.end(), [&str](const auto &el) { return el == str; });
			ASSERT(it != m_freevars.end());
			return std::distance(m_freevars.begin(), it) + m_cellvars.size();
		}

		void add_stack_size(size_t size) { m_stack_size += size; }
	};

	using FunctionsMap = std::unordered_map<std::string, FunctionInfo>;

	std::string m_filename;
	std::vector<std::string> m_argv;
	FunctionInfo m_module;
	FunctionsMap m_function_map;
	std::optional<std::reference_wrapper<typename FunctionsMap::value_type>> m_current_function;
	std::stack<decltype(LinearScanRegisterAllocation::value2mem_map)> m_register_mapping;
	std::stack<size_t> m_current_operation_index;
	std::stack<std::vector<Block *>> m_sorted_blocks;
	mlir::func::FuncOp m_parent_fn;
	InstructionSourceLocation m_current_op_location{};

	bool m_writing_to_module{ true };

	FunctionInfo &current_function()
	{
		if (m_writing_to_module) { return m_module; }
		ASSERT(m_current_function.has_value());
		return m_current_function->get().second;
	}

	struct BlockLabel
	{
		mlir::Block *m_block;
		std::shared_ptr<Label> m_label;
	};

	struct BlockOffset
	{
		mlir::Block *m_block;
		// offset from function start
		size_t m_offset;
	};

	std::vector<BlockLabel> m_block_labels;
	std::vector<BlockOffset> m_block_offsets;

	PythonBytecodeEmitter() = default;

	template<typename InstructionT, typename... Args> void emit(Args &&...args)
	{
		auto instruction = std::make_unique<InstructionT>(std::forward<Args>(args)...);
		record_location_for_next_instruction();
		current_function().instructions.push_back(std::move(instruction));
	}

	void record_location_for_next_instruction()
	{
		auto &locations = current_function().instruction_locations;
		if (!locations.empty() && locations.back().line == m_current_op_location.line
			&& locations.back().column == m_current_op_location.column) {
			return;
		}
		const auto next_instruction_index =
			static_cast<uint32_t>(current_function().instructions.size());
		locations.emplace_back(
			next_instruction_index, m_current_op_location.line, m_current_op_location.column);
	}

	void capture_op_location(mlir::Operation &op)
	{
		if (auto loc = mlir::dyn_cast<mlir::FileLineColLoc>(op.getLoc())) {
			m_current_op_location.line = loc.getLine() + 1;
			m_current_op_location.column = loc.getColumn();
		}
	}

	size_t add_name(std::string_view str) { return current_function().add_name(str); }

	size_t get_cell_index(std::string_view str) { return current_function().get_cell_index(str); }

	size_t add_const(::py::Value value) { return current_function().add_const(std::move(value)); }

	::py::Value get_value(mlir::Attribute attr) const
	{
		ASSERT(attr);
		::py::Value value =
			llvm::TypeSwitch<mlir::Attribute, ::py::Value>(attr)
				.Case<FloatAttr>([](FloatAttr f) { return ::py::Number{ f.getValueAsDouble() }; })
				.Case<BoolAttr>([](BoolAttr b) { return ::py::NameConstant{ b.getValue() }; })
				.Case<UnitAttr>([](UnitAttr) { return ::py::NameConstant{ ::py::NoneType{} }; })
				.Case<StringAttr>([](StringAttr str) { return ::py::String{ str.str() }; })
				.Case<IntegerAttr>([](IntegerAttr int_attr) {
					const auto &int_value = int_attr.getAPSInt();
					::py::BigIntType big_int_value{};
					if (int_value.isZero()) {
						big_int_value.get_mpz_t()->_mp_size = 0;
					} else {
						mpz_init2(big_int_value.get_mpz_t(), int_value.getBitWidth());
						big_int_value.get_mpz_t()->_mp_size =
							int_value.getNumWords() * (int_value.isNegative() ? -1 : 1);
						std::copy_n(int_value.getRawData(),
							int_value.getNumWords(),
							big_int_value.get_mpz_t()->_mp_d);
					}
					return ::py::Number{ std::move(big_int_value) };
				})
				.Case<DenseIntElementsAttr>([](DenseIntElementsAttr bytes_attr) {
					std::vector<std::byte> bytes;
					for (const auto &el : bytes_attr) {
						ASSERT(el.isIntN(8));
						bytes.push_back(std::byte{ *bit_cast<const uint8_t *>(el.getRawData()) });
					}
					return ::py::Bytes{ std::move(bytes) };
				})
				.Case<ArrayAttr>([this](ArrayAttr arr) {
					std::vector<::py::Value> elements;
					elements.reserve(arr.size());
					for (const auto &el : arr) { elements.push_back(get_value(el)); }
					return ::py::Tuple{ std::move(elements) };
				})
				.Default([](auto) {
					TODO();
					return ::py::NameConstant{ ::py::NoneType{} };
				});
		return value;
	};

	Register get_register(const mlir::Value &value) const
	{
		const auto mem = m_register_mapping.top().at(value);
		ASSERT(std::holds_alternative<LinearScanRegisterAllocation::Reg>(mem));
		const auto reg = std::get<LinearScanRegisterAllocation::Reg>(mem).idx;
		ASSERT(reg <= std::numeric_limits<Register>::max());

		return reg;
	}

	Register get_register(mlir::Operation *producing_operation,
		size_t produced_operation_index) const
	{
		const auto mem = m_register_mapping.top().at(
			std::make_pair(producing_operation, produced_operation_index));
		ASSERT(std::holds_alternative<LinearScanRegisterAllocation::Reg>(mem));
		const auto reg = std::get<LinearScanRegisterAllocation::Reg>(mem).idx;
		ASSERT(reg <= std::numeric_limits<Register>::max());

		return reg;
	}

	Register get_name_idx(StringRef name) const
	{
		auto names = const_cast<mlir::func::FuncOp &>(m_parent_fn).getOperation()->getAttr("names");
		ASSERT(names);
		auto names_array = mlir::cast<mlir::ArrayAttr>(names);
		auto it =
			std::find_if(names_array.begin(), names_array.end(), [&name](mlir::Attribute attr) {
				return mlir::cast<mlir::StringAttr>(attr).getValue() == name;
			});
		ASSERT(it != names_array.end());
		const auto idx = std::distance(names_array.begin(), it);
		ASSERT(idx <= std::numeric_limits<Register>::max());
		return idx;
	}

	Register get_local_idx(StringRef name) const
	{
		ASSERT(!m_writing_to_module);
		const auto &varnames = m_current_function->get().second.m_varnames;
		auto it = std::find_if(varnames.begin(), varnames.end(), [&name](const std::string &el) {
			return el == name;
		});
		ASSERT(it != varnames.end());
		const auto &cellvars = m_current_function->get().second.m_cellvars;
		const auto idx =
			std::count_if(varnames.begin(), it, [&cellvars](const std::string &varname) {
				return std::find(cellvars.begin(), cellvars.end(), varname) == cellvars.end();
			});
		ASSERT(idx <= std::numeric_limits<Register>::max());
		return idx;
	}

	void enter_function_op(mlir::func::FuncOp op)
	{
		auto extract_str_array = [&op](std::string_view array_name,
									 std::vector<std::string> &func_array) {
			auto attr = op.getOperation()->getAttr(array_name);
			if (attr) {
				auto array = mlir::cast<mlir::ArrayAttr>(attr);
				func_array.reserve(array.size());
				std::transform(array.begin(),
					array.end(),
					std::back_inserter(func_array),
					[](mlir::Attribute attr) {
						return mlir::cast<mlir::StringAttr>(attr).getValue().str();
					});
			}
		};

		extract_str_array("locals", current_function().m_varnames);
		extract_str_array("names", current_function().m_names);
		extract_str_array("cellvars", current_function().m_cellvars);
		extract_str_array("freevars", current_function().m_freevars);
		current_function().m_cell2arg = std::vector<size_t>(
			current_function().m_cellvars.size(), op.getFunctionType().getNumInputs());
		if (op.getAllArgAttrs()
			&& std::any_of(op.getAllArgAttrs().begin(),
				op.getAllArgAttrs().end(),
				[](mlir::Attribute arg_attr) {
					auto vararg = mlir::cast<mlir::DictionaryAttr>(arg_attr).getAs<mlir::BoolAttr>(
						"llvm.vararg");
					return vararg && vararg.getValue();
				})) {
			current_function().set_varargs();
		}

		if (op.getAllArgAttrs()
			&& std::any_of(op.getAllArgAttrs().begin(),
				op.getAllArgAttrs().end(),
				[](mlir::Attribute arg_attr) {
					auto vararg = mlir::cast<mlir::DictionaryAttr>(arg_attr).getAs<mlir::BoolAttr>(
						"llvm.kwarg");
					return vararg && vararg.getValue();
				})) {
			current_function().set_kwargs();
		}

		if (op.getAllArgAttrs()) {
			auto kwonlyarg_count = std::count_if(op.getAllArgAttrs().begin(),
				op.getAllArgAttrs().end(),
				[](mlir::Attribute arg_attr) {
					auto vararg = mlir::cast<mlir::DictionaryAttr>(arg_attr).getAs<mlir::BoolAttr>(
						"llvm.kwonlyarg");
					return vararg && vararg.getValue();
				});
			current_function().set_kwonlyarg_count(kwonlyarg_count);
		}

		if (auto is_generator = op->getAttrOfType<mlir::BoolAttr>("is_generator");
			is_generator && is_generator.getValue()) {
			current_function().set_is_generator();
		}

		if (auto is_async = op->getAttrOfType<mlir::BoolAttr>("async");
			is_async && is_async.getValue()) {
			current_function().set_is_async();
		}

		if (auto is_class = op->getAttrOfType<mlir::BoolAttr>("is_class");
			is_class && is_class.getValue()) {
			current_function().set_is_class();
		}

		ASSERT(!m_writing_to_module || op.getFunctionType().getNumInputs() == 0);
		if (!m_writing_to_module) {
			current_function().m_arg_count =
				op.getFunctionType().getNumInputs() - current_function().m_kwonly_arg_count
				- current_function().m_flags.is_set(CodeFlags::Flag::VARARGS)
				- current_function().m_flags.is_set(CodeFlags::Flag::VARKEYWORDS);
			for (const auto &arg :
				llvm::enumerate(current_function().m_varnames
								| std::ranges::views::take(current_function().m_arg_count))) {
				const auto &arg_idx = arg.index();
				const auto &arg_name = arg.value();
				if (auto it = std::ranges::find(current_function().m_cellvars, arg_name);
					it != current_function().m_cellvars.end()) {
					const auto cell_idx = std::distance(current_function().m_cellvars.begin(), it);
					current_function().m_cell2arg[cell_idx] = arg_idx;
				}
			}

			if (current_function().m_flags.is_set(CodeFlags::Flag::VARARGS)) {
				auto vararg_attr_it = std::find_if(op.getAllArgAttrs().begin(),
					op.getAllArgAttrs().end(),
					[](mlir::Attribute arg_attr) {
						auto vararg =
							mlir::cast<mlir::DictionaryAttr>(arg_attr).getAs<mlir::BoolAttr>(
								"llvm.vararg");
						return vararg && vararg.getValue();
					});
				auto arg_name = mlir::cast<mlir::DictionaryAttr>(*vararg_attr_it)
									.getAs<mlir::StringAttr>("llvm.name")
									.getValue();
				if (auto it = std::ranges::find(current_function().m_cellvars, arg_name);
					it != current_function().m_cellvars.end()) {
					const auto cell_idx = std::distance(current_function().m_cellvars.begin(), it);
					current_function().m_cell2arg[cell_idx] = current_function().m_arg_count;
				}
			}

			if (current_function().m_flags.is_set(CodeFlags::Flag::VARKEYWORDS)) {
				auto kwarg_attr_it = std::find_if(op.getAllArgAttrs().begin(),
					op.getAllArgAttrs().end(),
					[](mlir::Attribute arg_attr) {
						auto kwarg =
							mlir::cast<mlir::DictionaryAttr>(arg_attr).getAs<mlir::BoolAttr>(
								"llvm.kwarg");
						return kwarg && kwarg.getValue();
					});
				auto arg_name = mlir::cast<mlir::DictionaryAttr>(*kwarg_attr_it)
									.getAs<mlir::StringAttr>("llvm.name")
									.getValue();
				if (auto it = std::ranges::find(current_function().m_cellvars, arg_name);
					it != current_function().m_cellvars.end()) {
					const auto cell_idx = std::distance(current_function().m_cellvars.begin(), it);
					current_function().m_cell2arg[cell_idx] =
						current_function().m_arg_count
						+ current_function().m_flags.is_set(CodeFlags::Flag::VARARGS);
				}
			}
		}
		m_parent_fn = op;
	}

	const std::string &filename() const { return m_filename; }
	const std::vector<std::string> &argv() const { return m_argv; }

	template<typename OpType> LogicalResult emitOperation(OpType &op);

	void push(Register value);
};

template<> LogicalResult PythonBytecodeEmitter::emitOperation(Operation &op)
{
	capture_op_location(op);
	return llvm::TypeSwitch<Operation *, LogicalResult>(&op)
		// Builtin ops.
		.Case<mlir::ModuleOp>([this](auto op) {
			m_current_operation_index.push(0);
			return emitOperation(op);
		})
		.Case<mlir::emitpybytecode::ConstantOp,
			mlir::emitpybytecode::StoreFastOp,
			mlir::emitpybytecode::StoreGlobalOp,
			mlir::emitpybytecode::StoreNameOp,
			mlir::emitpybytecode::StoreDerefOp,
			mlir::emitpybytecode::LoadEllipsisOp,
			mlir::emitpybytecode::LoadFastOp,
			mlir::emitpybytecode::LoadNameOp,
			mlir::emitpybytecode::LoadGlobalOp,
			mlir::emitpybytecode::LoadClosureOp,
			mlir::emitpybytecode::LoadDerefOp,
			mlir::emitpybytecode::DeleteFastOp,
			mlir::emitpybytecode::DeleteNameOp,
			mlir::emitpybytecode::DeleteGlobalOp,
			mlir::emitpybytecode::DeleteDerefOp,
			mlir::emitpybytecode::UnpackSequenceOp,
			mlir::emitpybytecode::UnpackExpandOp>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::BinaryOp,
			mlir::emitpybytecode::UnaryOp,
			mlir::emitpybytecode::InplaceOp>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::FunctionCallOp,
			mlir::emitpybytecode::FunctionCallExOp,
			mlir::emitpybytecode::FunctionCallWithKeywordsOp>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::func::FuncOp>([this](auto op) {
			auto prev = m_parent_fn;
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			m_parent_fn = prev;
			return success();
		})
		.Case<mlir::emitpybytecode::MakeFunction>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::func::ReturnOp>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::Yield,
			mlir::emitpybytecode::YieldFrom,
			mlir::emitpybytecode::YieldFromIter>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::cf::BranchOp,
			mlir::emitpybytecode::JumpIfFalse,
			mlir::emitpybytecode::JumpIfNotException>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::Compare>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::LoadAssertionError,
			mlir::emitpybytecode::RaiseVarargs,
			mlir::emitpybytecode::ReRaiseOp>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::BuildDict,
			mlir::emitpybytecode::DictUpdate,
			mlir::emitpybytecode::DictAdd,
			mlir::emitpybytecode::BuildTuple,
			mlir::emitpybytecode::BuildSet,
			mlir::emitpybytecode::SetAdd,
			mlir::emitpybytecode::SetUpdate,
			mlir::emitpybytecode::BuildList,
			mlir::emitpybytecode::ListExtend,
			mlir::emitpybytecode::ListAppend,
			mlir::emitpybytecode::ListToTuple,
			mlir::emitpybytecode::BuildSlice,
			mlir::emitpybytecode::BuildString,
			mlir::emitpybytecode::FormatValue>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::LoadAttribute, mlir::emitpybytecode::LoadMethod>(
			[this](auto op) {
				if (emitOperation(op).failed()) { return failure(); };
				m_current_operation_index.top()++;
				return success();
			})
		.Case<mlir::emitpybytecode::BinarySubscript,
			mlir::emitpybytecode::StoreSubscript,
			mlir::emitpybytecode::DeleteSubscript>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			m_current_operation_index.top()++;
			return success();
		})
		.Case<mlir::emitpybytecode::StoreAttribute, emitpybytecode::DeleteAttribute>(
			[this](auto op) {
				if (emitOperation(op).failed()) { return failure(); };
				return success();
			})
		.Case<mlir::emitpybytecode::Push, mlir::emitpybytecode::Pop, mlir::emitpybytecode::Move>(
			[this](auto op) {
				if (emitOperation(op).failed()) { return failure(); };
				return success();
			})
		.Case<mlir::emitpybytecode::SetupExceptionHandle,
			mlir::emitpybytecode::SetupWith,
			mlir::emitpybytecode::WithExceptStart,
			mlir::emitpybytecode::ClearExceptionState,
			mlir::emitpybytecode::LeaveExceptionHandle>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			return success();
		})
		.Case<mlir::emitpybytecode::ImportName,
			mlir::emitpybytecode::ImportFrom,
			mlir::emitpybytecode::ImportAll>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			return success();
		})
		.Case<mlir::emitpybytecode::CastToBool>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			return success();
		})
		.Case<mlir::emitpybytecode::ForIter, mlir::emitpybytecode::GetIter>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			return success();
		})
		.Case<mlir::emitpybytecode::LoadBuildClass>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			return success();
		})
		.Case<mlir::emitpybytecode::ForIter, mlir::emitpybytecode::GetAwaitableOp>([this](auto op) {
			if (emitOperation(op).failed()) { return failure(); };
			return success();
		})
		.Default([this](Operation *op) {
			llvm::outs() << "Operation " << op->getName() << " not implemented\n";
			return failure();
		});
}

void PythonBytecodeEmitter::push(Register value)
{
	if (m_current_function.has_value()) { m_current_function->get().second.add_stack_size(1); }
	emit<Push>(value);
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadBuildClass &op)
{
	emit<LoadBuildClass>(get_register(op.getClassBuilder()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ImportName &op)
{
	emit<ImportName>(get_register(op.getModule()),
		add_name(op.getName()),
		get_register(op.getFromList()),
		get_register(op.getLevel()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ImportFrom &op)
{
	emit<ImportFrom>(
		get_register(op.getObject()), add_name(op.getName()), get_register(op.getModule()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ImportAll &op)
{
	emit<ImportStar>(get_register(op.getModule()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::CastToBool &op)
{
	emit<ToBool>(get_register(op.getOutput()), get_register(op.getValue()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ForIter &op)
{
	auto exit_label =
		m_block_labels.emplace_back(op.getContinuation(), std::make_shared<Label>("", 0)).m_label;
	auto body_label =
		m_block_labels.emplace_back(op.getBody(), std::make_shared<Label>("", 0)).m_label;

	emit<ForIter>(get_register(op, 0),
		get_register(op.getIterator()),
		std::move(body_label),
		std::move(exit_label));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::GetIter &op)
{
	emit<GetIter>(get_register(op.getIterator()), get_register(op.getIterable()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::SetupExceptionHandle &op)
{
	auto &handler_label =
		m_block_labels.emplace_back(op.getHandler(), std::make_shared<Label>("", 0));
	emit<SetupExceptionHandling>(handler_label.m_label);
	auto &body_label = m_block_labels.emplace_back(op.getBody(), std::make_shared<Label>("", 0));
	emit<Jump>(body_label.m_label);
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::SetupWith &op)
{
	auto &handler_label =
		m_block_labels.emplace_back(op.getHandler(), std::make_shared<Label>("", 0));
	emit<SetupWith>(handler_label.m_label);
	auto &body_label = m_block_labels.emplace_back(op.getBody(), std::make_shared<Label>("", 0));
	emit<Jump>(body_label.m_label);
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::WithExceptStart &op)
{
	emit<WithExceptStart>(get_register(op.getOutput()), get_register(op.getExitMethod()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LeaveExceptionHandle &)
{
	emit<LeaveExceptionHandling>();
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ClearExceptionState &)
{
	emit<ClearExceptionState>();
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::func::FuncOp &op)
{
	// Register allocation runs inline here rather than as an MLIR pass.
	// Plan step 19 attempted to wrap LinearScanRegisterAllocation as an
	// OperationPass<func::FuncOp> with the emitter recovering the value ->
	// register map by re-running analyse() on the now-spilled IR. That
	// failed in integration tests: two analyse() calls on identical IR
	// can produce different assignments (a register that should hold a
	// set ended up holding a range in set.py and friends). The algorithm
	// has some state-dependent or non-deterministic tie-breaking that
	// makes "spilled IR + fresh analyse" diverge from "iterative analyse
	// that performed the spilling". Lifting allocation into a pass
	// therefore requires plumbing the assignment from pass to emitter -
	// e.g. encoding it as op attributes or via a side-channel keyed by
	// FuncOp identity. Tracked separately.
	LinearScanRegisterAllocation register_allocation{};
	register_allocation.analyse(op, mlir::OpBuilder{ op.getContext() });
	m_register_mapping.push(register_allocation.value2mem_map);

	std::string prev_func;
	const auto entry_attr = op->getAttrOfType<mlir::BoolAttr>("is_module_entry");
	const bool is_module_entry = entry_attr && entry_attr.getValue();
	if (!is_module_entry) {
		auto current_func = m_function_map.emplace(op.getSymName(), FunctionInfo{});
		prev_func = m_current_function.has_value() ? m_current_function->get().first : "";
		m_current_function = *current_func.first;
	}
	FunctionInfo &function_info = is_module_entry ? m_module : m_current_function->get().second;

	const bool prev_writing_to_module = std::exchange(m_writing_to_module, is_module_entry);

	m_current_operation_index.push(0);

	auto &region = op.getRegion();

	enter_function_op(op);

	m_sorted_blocks.push(sortBlocks(region));
	for (size_t op_idx = 0; auto *block : m_sorted_blocks.top()) {
		m_block_offsets.emplace_back(block, op_idx);
		if (!sortTopologically(block)) { std::abort(); }
		const auto start = function_info.instructions.size();
		for (auto &op : block->getOperations()) {
			if (failed(emitOperation(op))) { return failure(); }
		}
		op_idx += function_info.instructions.size() - start;
	}

	if (!prev_writing_to_module) {
		m_current_function = *m_function_map.find(prev_func);
	} else {
		m_current_function = {};
	}
	m_register_mapping.pop();
	m_sorted_blocks.pop();
	m_writing_to_module = prev_writing_to_module;

	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::MakeFunction &op)
{
	for (const auto &default_ : op.getDefaults()) { push(get_register(default_)); }
	for (const auto &default_ : op.getKwDefaults()) { push(get_register(default_)); }

	const size_t defaults_size = op.getDefaults().size();
	const size_t kw_defaults_size = op.getKwDefaults().size();
	auto captures = op.getCaptures() ? get_register(op.getCaptures()) : std::optional<Register>{};
	emit<MakeFunction>(
		get_register(op), get_register(op.getSymName()), defaults_size, kw_defaults_size, captures);

	for (size_t i = 0; i < defaults_size + kw_defaults_size; ++i) { emit<Pop>(); }

	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::cf::BranchOp &op)
{
	auto *bb = op.getDest();
	auto &label = m_block_labels.emplace_back(bb, std::make_shared<Label>("", 0));
	emit<Jump>(label.m_label);
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadAssertionError &op)
{
	emit<LoadAssertionError>(get_register(op.getOutput()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::DictUpdate &op)
{
	emit<DictUpdate>(get_register(op.getDict()), get_register(op.getMappable()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::DictAdd &op)
{
	emit<DictAdd>(
		get_register(op.getDict()), get_register(op.getKey()), get_register(op.getValue()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BuildDict &op)
{
	ASSERT(op.getKeys().size() == op.getValues().size());
	for (const auto &key : op.getKeys()) { push(get_register(key)); }
	for (const auto &value : op.getValues()) { push(get_register(value)); }
	emit<BuildDict>(get_register(op.getOutput()), op.getKeys().size());
	for (size_t i = 0; i < op.getKeys().size() + op.getValues().size(); ++i) { emit<Pop>(); }
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BuildList &op)
{
	for (const auto &el : op.getElements()) { push(get_register(el)); }
	emit<BuildList>(get_register(op.getOutput()), op.getElements().size());
	for (size_t i = 0; i < op.getElements().size(); ++i) { emit<Pop>(); }
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ListExtend &op)
{
	emit<ListExtend>(get_register(op.getList()), get_register(op.getIterable()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ListAppend &op)
{
	emit<ListAppend>(get_register(op.getList()), get_register(op.getValue()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ListToTuple &op)
{
	emit<ListToTuple>(get_register(op.getTuple()), get_register(op.getList()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BuildTuple &op)
{
	for (const auto &el : op.getElements()) { push(get_register(el)); }
	emit<BuildTuple>(get_register(op.getOutput()), op.getElements().size());
	for (size_t i = 0; i < op.getElements().size(); ++i) { emit<Pop>(); }
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BuildSet &op)
{
	for (const auto &el : op.getElements()) { push(get_register(el)); }
	emit<BuildSet>(get_register(op.getOutput()), op.getElements().size());
	for (size_t i = 0; i < op.getElements().size(); ++i) { emit<Pop>(); }
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BuildString &op)
{
	for (const auto &el : op.getElements()) { push(get_register(el)); }
	emit<BuildString>(get_register(op.getOutput()), op.getElements().size());
	for (size_t i = 0; i < op.getElements().size(); ++i) { emit<Pop>(); }
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::FormatValue &op)
{
	emit<FormatValue>(
		get_register(op.getOutput()), get_register(op.getValue()), op.getConversion());
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BuildSlice &op)
{
	if (op.getStep()) {
		emit<BuildSlice>(get_register(op.getOutput()),
			get_register(op.getLower()),
			get_register(op.getUpper()),
			get_register(op.getStep()));
	} else {
		emit<BuildSlice>(
			get_register(op.getOutput()), get_register(op.getLower()), get_register(op.getUpper()));
	}
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::SetAdd &op)
{
	emit<SetAdd>(get_register(op.getSet()), get_register(op.getValue()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::SetUpdate &op)
{
	emit<SetUpdate>(get_register(op.getSet()), get_register(op.getIterable()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadAttribute &op)
{
	// LoadAttribute has side effects (can raise AttributeError, trigger descriptors)
	// so we must emit it even if the result is unused. Liveness analysis ensures
	// side-effecting operations always get register assignments.
	emit<LoadAttr>(
		get_register(op.getOutput()), get_register(op.getSelf()), add_name(op.getAttr()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadMethod &op)
{
	emit<LoadMethod>(
		get_register(op.getMethod()), get_register(op.getSelf()), add_name(op.getMethodName()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BinarySubscript &op)
{
	emit<BinarySubscript>(
		get_register(op.getOutput()), get_register(op.getSelf()), get_register(op.getSubscript()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::StoreSubscript &op)
{
	emit<StoreSubscript>(
		get_register(op.getSelf()), get_register(op.getSubscript()), get_register(op.getValue()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::DeleteSubscript &op)
{
	emit<DeleteSubscript>(get_register(op.getSelf()), get_register(op.getSubscript()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::StoreAttribute &op)
{
	emit<StoreAttr>(
		get_register(op.getSelf()), get_register(op.getValue()), get_name_idx(op.getAttr()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::DeleteAttribute &op)
{
	emit<DeleteAttr>(get_register(op.getSelf()), add_name(op.getAttr()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::RaiseVarargs &op)
{
	if (auto cause = op.getCause()) {
		emit<RaiseVarargs>(get_register(op.getException()), get_register(cause));
	} else {
		emit<RaiseVarargs>(get_register(op.getException()));
	}
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ReRaiseOp &op)
{
	emit<ReRaise>();
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::JumpIfFalse &op)
{
	const auto &sorted = m_sorted_blocks.top();
	auto this_block = std::find(sorted.begin(), sorted.end(), op.getOperation()->getBlock());
	ASSERT(this_block != sorted.end());
	auto next_it = this_block + 1;
	mlir::Block *next_block = next_it == sorted.end() ? nullptr : *next_it;

	if (next_block == op.getTrueDest()) {
		auto &label =
			m_block_labels.emplace_back(op.getFalseDest(), std::make_shared<Label>("", 0));
		emit<JumpIfFalse>(get_register(op.getCond()), label.m_label);
	} else if (next_block == op.getFalseDest()) {
		auto &label = m_block_labels.emplace_back(op.getTrueDest(), std::make_shared<Label>("", 0));
		emit<JumpIfTrue>(get_register(op.getCond()), label.m_label);
	} else {
		// Neither successor falls through: emit a conditional jump to the false
		// dest followed by an unconditional jump to the true dest.
		auto &false_label =
			m_block_labels.emplace_back(op.getFalseDest(), std::make_shared<Label>("", 0));
		emit<JumpIfFalse>(get_register(op.getCond()), false_label.m_label);
		auto &true_label =
			m_block_labels.emplace_back(op.getTrueDest(), std::make_shared<Label>("", 0));
		emit<Jump>(true_label.m_label);
	}
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::JumpIfNotException &op)
{
	const auto &sorted = m_sorted_blocks.top();
	auto this_block = std::find(sorted.begin(), sorted.end(), op.getOperation()->getBlock());
	ASSERT(this_block != sorted.end());
	auto next_it = this_block + 1;
	mlir::Block *next_block = next_it == sorted.end() ? nullptr : *next_it;

	if (next_block == op.getTrueDest()) {
		auto &label =
			m_block_labels.emplace_back(op.getFalseDest(), std::make_shared<Label>("", 0));
		emit<JumpIfExceptionMatch>(get_register(op.getObjectType()), label.m_label);
	} else if (next_block == op.getFalseDest()) {
		auto &label = m_block_labels.emplace_back(op.getTrueDest(), std::make_shared<Label>("", 0));
		emit<JumpIfNotExceptionMatch>(get_register(op.getObjectType()), label.m_label);
	} else {
		auto &true_label =
			m_block_labels.emplace_back(op.getTrueDest(), std::make_shared<Label>("", 0));
		emit<JumpIfNotExceptionMatch>(get_register(op.getObjectType()), true_label.m_label);
		auto &false_label =
			m_block_labels.emplace_back(op.getFalseDest(), std::make_shared<Label>("", 0));
		emit<Jump>(false_label.m_label);
	}

	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::ConstantOp &op)
{
	return llvm::TypeSwitch<mlir::Attribute, LogicalResult>(op.getValue())
		.Case<FloatAttr,
			BoolAttr,
			UnitAttr,
			StringAttr,
			IntegerAttr,
			DenseIntElementsAttr,
			ArrayAttr>([this, &op](auto attr) {
			const auto idx = add_const(get_value(attr));
			emit<LoadConst>(get_register(op.getResult()), idx);
			return success();
		})
		.Default([](auto) {
			TODO();
			return failure();
		});
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadEllipsisOp &op)
{
	const auto idx = add_const(::py::Ellipsis{});
	emit<LoadConst>(get_register(op.getResult()), idx);
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::StoreFastOp &op)
{
	emit<StoreFast>(get_local_idx(op.getName()), get_register(op.getOperand()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::StoreGlobalOp &op)
{
	emit<StoreGlobal>(get_name_idx(op.getName()), get_register(op.getOperand()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::StoreNameOp &op)
{
	emit<StoreName>(op.getName().str(), get_register(op.getOperand()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::StoreDerefOp &op)
{
	emit<StoreDeref>(get_cell_index(op.getName().str()), get_register(op.getOperand()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::FunctionCallOp &op)
{
	const auto arg_size = op.getArgs().size();
	for (const auto &arg : op.getArgs()) { push(get_register(arg)); }
	emit<FunctionCall>(get_register(op.getCallee()), arg_size, 0);
	for (size_t i = 0; i < arg_size; ++i) { emit<Pop>(); }

	// CALL operations always put their result in r0.
	// If the register allocator assigned a different register, emit a MOVE.
	const auto result_reg = get_register(op.getOutput());
	if (result_reg != 0) {
		emit<Move>(result_reg, 0);// Move from r0 to allocated register
	}

	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(
	mlir::emitpybytecode::FunctionCallWithKeywordsOp &op)
{
	std::vector<Register> arg_registers;
	std::vector<Register> kwarg_registers;
	std::vector<Register> keywords_registers;

	arg_registers.reserve(op.getArgs().size());
	for (const auto &arg : op.getArgs()) { arg_registers.push_back(get_register(arg)); }

	kwarg_registers.reserve(op.getKwargs().size());
	keywords_registers.reserve(op.getKeywords().size());
	for (auto [keyword, value] :
		llvm::zip(op.getKeywords().getValues<mlir::StringRef>(), op.getKwargs())) {
		kwarg_registers.push_back(get_register(value));
		keywords_registers.push_back(add_name(keyword));
	}

	emit<FunctionCallWithKeywords>(get_register(op.getCallee()),
		std::move(arg_registers),
		std::move(kwarg_registers),
		std::move(keywords_registers));

	// CALL operations always put their result in r0.
	// If the register allocator assigned a different register, emit a MOVE.
	const auto result_reg = get_register(op.getOutput());
	if (result_reg != 0) {
		emit<Move>(result_reg, 0);// Move from r0 to allocated register
	}

	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::FunctionCallExOp &op)
{
	emit<FunctionCallEx>(get_register(op.getCallee()),
		op.getArgs() ? get_register(op.getArgs()) : Register{ 0 },
		op.getKwargs() ? get_register(op.getKwargs()) : Register{ 0 },
		op.getArgs() != nullptr,
		op.getKwargs() != nullptr);

	// CALL operations always put their result in r0.
	// If the register allocator assigned a different register, emit a MOVE.
	const auto result_reg = get_register(op.getOutput());
	if (result_reg != 0) {
		emit<Move>(result_reg, 0);// Move from r0 to allocated register
	}

	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::func::ReturnOp &op)
{
	ASSERT(op.getOperands().size() == 1);
	emit<ReturnValue>(get_register(op.getOperands().back()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::Yield &op)
{
	emit<YieldValue>(get_register(op.getValue()));
	// TODO: optimise away the YieldLoad when the received value is never used
	//       would have to happen during op lowering to EmitPythonBytecode dialect in order to avoid
	//       register allocation of `received`
	emit<YieldLoad>(get_register(op.getReceived()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::YieldFrom &op)
{
	emit<YieldFrom>(get_register(op.getReceived()),
		get_register(op.getIterator()),
		get_register(op.getValue()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::YieldFromIter &op)
{
	emit<GetYieldFromIter>(get_register(op.getIterator()), get_register(op.getIterable()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::Push &op)
{
	const auto src = op.getSrc();
	ASSERT(src <= std::numeric_limits<Register>::max());
	push(src);
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::Pop &op)
{
	const auto dst = op.getDst();
	ASSERT(dst <= std::numeric_limits<Register>::max());
	emit<Pop>(dst);
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::Move &op)
{
	const auto src = op.getSrc();
	ASSERT(src <= std::numeric_limits<Register>::max());

	const auto dst = op.getDst();
	ASSERT(dst <= std::numeric_limits<Register>::max());

	emit<Move>(dst, src);

	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::BinaryOp &op)
{
	emit<BinaryOperation>(get_register(op.getOutput()),
		get_register(op.getLhs()),
		get_register(op.getRhs()),
		static_cast<BinaryOperation::Operation>(op.getOperationType()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::UnaryOp &op)
{
	emit<Unary>(get_register(op.getOutput()),
		get_register(op.getInput()),
		static_cast<Unary::Operation>(op.getOperationType()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::InplaceOp &op)
{
	emit<InplaceOp>(get_register(op.getDst()),
		get_register(op.getSrc()),
		static_cast<InplaceOp::Operation>(op.getOperationType()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadNameOp &op)
{
	emit<LoadName>(get_register(op.getOutput()), op.getName().str());
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadFastOp &op)
{
	emit<LoadFast>(get_register(op.getOutput()), get_local_idx(op.getName()), op.getName().str());
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadGlobalOp &op)
{
	emit<LoadGlobal>(get_register(op.getOutput()), add_name(op.getName()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadDerefOp &op)
{
	emit<LoadDeref>(get_register(op.getOutput()), get_cell_index(op.getName()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::LoadClosureOp &op)
{
	emit<LoadClosure>(get_register(op.getOutput()), get_cell_index(op.getName()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::DeleteNameOp &op)
{
	emit<DeleteName>(add_const(get_value(op.getNameAttr())));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::DeleteFastOp &op)
{
	emit<DeleteFast>(get_local_idx(op.getName()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::DeleteGlobalOp &op)
{
	emit<DeleteGlobal>(add_const(get_value(op.getNameAttr())));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::DeleteDerefOp &op)
{
	emit<DeleteDeref>(get_cell_index(op.getName()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::UnpackSequenceOp &op)
{
	std::vector<Register> unpacked_values;
	for (const auto &el : op.getUnpackedValues()) { unpacked_values.push_back(get_register(el)); }
	emit<UnpackSequence>(unpacked_values, get_register(op.getIterable()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::UnpackExpandOp &op)
{
	std::vector<Register> unpacked_values;
	for (const auto &el : op.getUnpackedValues()) { unpacked_values.push_back(get_register(el)); }
	emit<UnpackExpand>(unpacked_values, get_register(op.getRest()), get_register(op.getIterable()));
	return success();
}

template<>
LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::GetAwaitableOp &op)
{
	emit<GetAwaitable>(get_register(op.getIterator()), get_register(op.getIterable()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::emitpybytecode::Compare &op)
{
	emit<CompareOperation>(get_register(op.getOutput()),
		get_register(op.getLhs()),
		get_register(op.getRhs()),
		static_cast<CompareOperation::Comparisson>(op.getPredicate()));
	return success();
}

template<> LogicalResult PythonBytecodeEmitter::emitOperation(mlir::ModuleOp &module_)
{
	m_filename = mlir::cast<mlir::FileLineColLoc>(module_.getLoc()).getFilename().str();
	auto argv = module_->getAttr("llvm.argv");
	ASSERT(argv);
	auto argv_array = mlir::cast<mlir::ArrayAttr>(argv);
	m_argv.reserve(argv_array.size());
	for (const auto &argv_ : argv_array) {
		m_argv.push_back(mlir::cast<mlir::StringAttr>(argv_).str());
	}
	auto &module_region = module_.getBodyRegion();
	ASSERT(module_region.getBlocks().size() == 1);
	auto fn = std::find_if(module_region.getBlocks().back().getOperations().begin(),
		module_region.getBlocks().back().getOperations().end(),
		[](mlir::Operation &op) {
			auto fn = mlir::cast<mlir::func::FuncOp>(op);
			if (!fn) { return false; }
			auto attr = fn->getAttrOfType<mlir::BoolAttr>("is_module_entry");
			return attr && attr.getValue();
		});
	ASSERT(fn != module_region.getBlocks().back().getOperations().end());
	m_parent_fn = mlir::cast<mlir::func::FuncOp>(*fn);

	if (failed(emitOperation(m_parent_fn))) { return failure(); }

	for (auto &op : module_region.getBlocks().back().getOperations()) {
		ASSERT(mlir::isa<mlir::func::FuncOp>(op));
		if (auto entry = op.getAttrOfType<mlir::BoolAttr>("is_module_entry");
			entry && entry.getValue()) {
			continue;
		}
		if (failed(emitOperation(op))) { return failure(); }
	}

	for (auto &block_label : m_block_labels) {
		auto *bb = block_label.m_block;
		auto &label = block_label.m_label;
		auto it = std::find_if(m_block_offsets.begin(),
			m_block_offsets.end(),
			[bb](const auto &el) { return el.m_block == bb; });
		ASSERT(it != m_block_offsets.end());
		label->set_position(it->m_offset);
	}

	size_t instruction_idx{ 0 };
	for (const auto &ins : m_module.instructions) { ins->relocate(instruction_idx++); }
	for (const auto &[_, func] : m_function_map) {
		instruction_idx = 0;
		for (const auto &ins : func.instructions) { ins->relocate(instruction_idx++); }
	}

	return success();
}


std::shared_ptr<Program> translateToPythonBytecode(Operation *op)
{
	if (mlir::verify(op, /*verifyRecursively*/ true).failed()) {
		std::cerr << "Invalid Python bytecode IR\n";
		return nullptr;
	}

	DialectRegistry registry;
	registry.insert<emitpybytecode::EmitPythonBytecodeDialect>();
	PythonBytecodeEmitter emitter;
	if (failed(emitter.emitOperation(*op))) { return nullptr; }

	FunctionBlocks func_blocks;
	InstructionVector instructions = std::move(emitter.m_module.instructions);
	{
		FunctionBlock fb_module{
			.metadata =
				FunctionMetaData{
					.function_name = "__main__",
					.register_count = codegen::kNumRegisters,
					.stack_size = emitter.m_module.m_stack_size,
					.names = std::move(emitter.m_module.m_names),
					.filename = emitter.m_filename,
					.consts = std::move(emitter.m_module.m_consts),
				},
			.blocks = std::move(instructions),
			.instruction_locations = std::move(emitter.m_module.instruction_locations),
		};
		func_blocks.functions.push_back(std::move(fb_module));
	}
	for (auto &&[name, fn] : emitter.m_function_map) {
		const auto stack_size = fn.m_varnames.size() + fn.m_stack_size;
		FunctionBlock fb{
			.metadata =
				FunctionMetaData{
					.function_name = name,
					.register_count = codegen::kNumRegisters,
					.stack_size = stack_size,
					.cellvars = std::move(fn.m_cellvars),
					.varnames = std::move(fn.m_varnames),
					.freevars = std::move(fn.m_freevars),
					.names = std::move(fn.m_names),
					.filename = emitter.m_filename,
					.arg_count = fn.m_arg_count,
					.kwonly_arg_count = fn.m_kwonly_arg_count,
					.cell2arg = std::move(fn.m_cell2arg),
					.consts = std::move(fn.m_consts),
					.flags = std::move(fn.m_flags),
				},
			.blocks = std::move(fn.instructions),
			.instruction_locations = std::move(fn.instruction_locations),
		};
		func_blocks.functions.push_back(std::move(fb));
	}
	return BytecodeProgram::create(std::move(func_blocks), emitter.filename(), emitter.argv());
}

}// namespace codegen
