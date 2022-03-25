#pragma once

#include "ast/AST.hpp"
#include "executable/Label.hpp"
#include "executable/bytecode/codegen/BytecodeGenerator.hpp"
#include "forward.hpp"
#include "interpreter/Interpreter.hpp"
#include "parser/Parser.hpp"
#include "utilities.hpp"
#include "vm/VM.hpp"

#include <sstream>

class Instruction : NonCopyable
{
  public:
	virtual ~Instruction() = default;
	virtual std::string to_string() const = 0;
	virtual void execute(VirtualMachine &, Interpreter &) const = 0;
	virtual void relocate(codegen::BytecodeGenerator &, size_t) = 0;
	// virtual std::vector<char> serialize() const = 0;
};

// std::vector<std::unique_ptr<Instruction>> deserialize(std::vector<char> instruction_stream);

class LoadConst final : public Instruction
{
	Register m_destination;
	size_t m_static_value_index;

  public:
	LoadConst(Register destination, size_t static_value_index)
		: m_destination(destination), m_static_value_index(static_value_index)
	{}
	~LoadConst() override {}
	std::string to_string() const final
	{
		return fmt::format("LOAD_CONST      r{:<3} s{:<3}", m_destination, m_static_value_index);
	}
	void execute(VirtualMachine &vm, Interpreter &interpreter) const final
	{
		ASSERT(vm.registers().has_value())
		ASSERT(vm.registers()->get().size() > m_destination)
		vm.reg(m_destination) = interpreter.execution_frame()->consts(m_static_value_index);
	}

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};

class Move final : public Instruction
{
	Register m_destination;
	Register m_source;

  public:
	Move(Register destination, Register source) : m_destination(destination), m_source(source) {}
	~Move() override {}
	std::string to_string() const final
	{
		return fmt::format("MOVE            r{:<3}  r{:<3}", m_destination, m_source);
	}
	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};


class Add : public Instruction
{
	Register m_destination;
	Register m_lhs;
	Register m_rhs;

  public:
	Add(Register dst, Register lhs, Register rhs) : m_destination(dst), m_lhs(lhs), m_rhs(rhs) {}
	~Add() override {}
	std::string to_string() const final
	{
		return fmt::format("ADD             r{:<3} r{:<3} r{:<3}", m_destination, m_lhs, m_rhs);
	}
	void execute(VirtualMachine &vm, Interpreter &interpreter) const final
	{
		const auto &lhs = vm.reg(m_lhs);
		const auto &rhs = vm.reg(m_rhs);
		if (auto result = py::add(lhs, rhs, interpreter)) {
			ASSERT(vm.registers().has_value())
			ASSERT(vm.registers()->get().size() > m_destination)
			vm.reg(m_destination) = *result;
		}
	}

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};


class Subtract : public Instruction
{
	Register m_destination;
	Register m_lhs;
	Register m_rhs;

  public:
	Subtract(Register dst, Register lhs, Register rhs) : m_destination(dst), m_lhs(lhs), m_rhs(rhs)
	{}
	~Subtract() override {}
	std::string to_string() const final
	{
		return fmt::format("SUB             r{:<3} r{:<3} r{:<3}", m_destination, m_lhs, m_rhs);
	}
	void execute(VirtualMachine &vm, Interpreter &interpreter) const final
	{
		const auto &lhs = vm.reg(m_lhs);
		const auto &rhs = vm.reg(m_rhs);
		if (auto result = py::subtract(lhs, rhs, interpreter)) {
			ASSERT(vm.registers().has_value())
			ASSERT(vm.registers()->get().size() > m_destination)
			vm.reg(m_destination) = *result;
		}
	}

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};

class Multiply : public Instruction
{
	Register m_destination;
	Register m_lhs;
	Register m_rhs;

  public:
	Multiply(Register dst, Register lhs, Register rhs) : m_destination(dst), m_lhs(lhs), m_rhs(rhs)
	{}
	~Multiply() override {}
	std::string to_string() const final
	{
		return fmt::format("MUL             r{:<3} r{:<3} r{:<3}", m_destination, m_lhs, m_rhs);
	}
	void execute(VirtualMachine &vm, Interpreter &interpreter) const final
	{
		const auto &lhs = vm.reg(m_lhs);
		const auto &rhs = vm.reg(m_rhs);
		if (auto result = py::multiply(lhs, rhs, interpreter)) {
			ASSERT(vm.registers().has_value())
			ASSERT(vm.registers()->get().size() > m_destination)
			vm.reg(m_destination) = *result;
		}
	}
	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};


class Exp : public Instruction
{
	Register m_destination;
	Register m_lhs;
	Register m_rhs;

  public:
	Exp(Register dst, Register lhs, Register rhs) : m_destination(dst), m_lhs(lhs), m_rhs(rhs) {}
	~Exp() override {}
	std::string to_string() const final
	{
		return fmt::format("EXP             r{:<3} r{:<3} r{:<3}", m_destination, m_lhs, m_rhs);
	}
	void execute(VirtualMachine &vm, Interpreter &interpreter) const final
	{
		const auto &lhs = vm.reg(m_lhs);
		const auto &rhs = vm.reg(m_rhs);
		if (auto result = py::exp(lhs, rhs, interpreter)) {
			ASSERT(vm.registers().has_value())
			ASSERT(vm.registers()->get().size() > m_destination)
			vm.reg(m_destination) = *result;
		}
	}

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};

class Modulo : public Instruction
{
	Register m_destination;
	Register m_lhs;
	Register m_rhs;

  public:
	Modulo(Register dst, Register lhs, Register rhs) : m_destination(dst), m_lhs(lhs), m_rhs(rhs) {}
	~Modulo() override {}
	std::string to_string() const final
	{
		return fmt::format("MODULO         r{:<3} r{:<3} r{:<3}", m_destination, m_lhs, m_rhs);
	}
	void execute(VirtualMachine &vm, Interpreter &interpreter) const final
	{
		const auto &lhs = vm.reg(m_lhs);
		const auto &rhs = vm.reg(m_rhs);
		if (auto result = py::modulo(lhs, rhs, interpreter)) {
			ASSERT(vm.registers().has_value())
			ASSERT(vm.registers()->get().size() > m_destination)
			vm.reg(m_destination) = *result;
		}
	}

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};


class LeftShift : public Instruction
{
	Register m_destination;
	Register m_lhs;
	Register m_rhs;

  public:
	LeftShift(Register dst, Register lhs, Register rhs) : m_destination(dst), m_lhs(lhs), m_rhs(rhs)
	{}
	~LeftShift() override {}
	std::string to_string() const final
	{
		return fmt::format("LSHIFT          r{:<3} r{:<3} r{:<3}", m_destination, m_lhs, m_rhs);
	}
	void execute(VirtualMachine &vm, Interpreter &interpreter) const final
	{
		const auto &lhs = vm.reg(m_lhs);
		const auto &rhs = vm.reg(m_rhs);
		if (auto result = py::lshift(lhs, rhs, interpreter)) {
			ASSERT(vm.registers().has_value())
			ASSERT(vm.registers()->get().size() > m_destination)
			vm.reg(m_destination) = *result;
		}
	}

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};


class LoadFast final : public Instruction
{
	Register m_destination;
	Register m_stack_index;
	const std::string m_object_name;

  public:
	LoadFast(Register destination, Register stack_index, std::string object_name)
		: m_destination(destination), m_stack_index(stack_index),
		  m_object_name(std::move(object_name))
	{}
	~LoadFast() override {}
	std::string to_string() const final
	{
		return fmt::format(
			"LOAD_FAST       r{:<3} {} (\"{}\")", m_destination, m_stack_index, m_object_name);
	}

	Register dst() const { return m_destination; }
	Register stack_index() const { return m_stack_index; }
	const std::string &object_name() const { return m_object_name; }

	void execute(VirtualMachine &vm, Interpreter &) const final
	{
		vm.reg(m_destination) = vm.stack_local(m_stack_index);
	}

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};

class StoreFast final : public Instruction
{
	Register m_stack_index;
	const std::string m_object_name;
	Register m_src;

  public:
	StoreFast(size_t stack_index, std::string object_name, Register src)
		: m_stack_index(stack_index), m_object_name(std::move(object_name)), m_src(src)
	{}
	~StoreFast() override {}
	std::string to_string() const final
	{
		return fmt::format(
			"STORE_FAST       {} (\"{}\") r{:<3}", m_stack_index, m_object_name, m_src);
	}

	void execute(VirtualMachine &vm, Interpreter &) const final
	{
		vm.stack_local(m_stack_index) = vm.reg(m_src);
	}

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};


class MakeFunction : public Instruction
{
	Register m_dst;
	std::string m_function_name;
	std::vector<Register> m_defaults;
	std::vector<std::optional<Register>> m_kw_defaults;
	std::optional<Register> m_captures_tuple;

  public:
	MakeFunction(Register dst,
		std::string function_name,
		std::vector<Register> defaults,
		std::vector<std::optional<Register>> kw_defaults,
		std::optional<Register> captures_tuple)
		: m_dst(dst), m_function_name(std::move(function_name)), m_defaults(std::move(defaults)),
		  m_kw_defaults(std::move(kw_defaults)), m_captures_tuple(std::move(captures_tuple))
	{}

	std::string to_string() const final
	{
		return fmt::format("MAKE_FUNCTION   ({})", m_function_name);
	}

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};


class JumpIfFalse final : public Instruction
{
	Register m_test_register;
	std::shared_ptr<Label> m_label;
	std::optional<int32_t> m_offset;

  public:
	JumpIfFalse(Register test_register, std::shared_ptr<Label> label)
		: m_test_register(test_register), m_label(std::move(label))
	{}

	std::string to_string() const final
	{
		return fmt::format("JUMP_IF_FALSE   position: {}", m_label->position());
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final;
};

class Jump final : public Instruction
{
	std::shared_ptr<Label> m_label;
	std::optional<int32_t> m_offset;

  public:
	Jump(std::shared_ptr<Label> label) : m_label(std::move(label)) {}
	std::string to_string() const final
	{
		return fmt::format("JUMP            position: {}", m_label->position());
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final;
};

class Equal final : public Instruction
{
	Register m_dst;
	Register m_lhs;
	Register m_rhs;

  public:
	Equal(Register dst, Register lhs, Register rhs) : m_dst(dst), m_lhs(lhs), m_rhs(rhs) {}

	std::string to_string() const final
	{
		return fmt::format("EQUAL            r{:<3} r{:<3} r{:<3}", m_dst, m_lhs, m_rhs);
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};

class LessThanEquals final : public Instruction
{
	Register m_dst;
	Register m_lhs;
	Register m_rhs;

  public:
	LessThanEquals(Register dst, Register lhs, Register rhs) : m_dst(dst), m_lhs(lhs), m_rhs(rhs) {}

	std::string to_string() const final
	{
		return fmt::format("LESS_THAN_EQ    r{:<3} r{:<3} r{:<3}", m_dst, m_lhs, m_rhs);
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};

class LessThan final : public Instruction
{
	Register m_dst;
	Register m_lhs;
	Register m_rhs;

  public:
	LessThan(Register dst, Register lhs, Register rhs) : m_dst(dst), m_lhs(lhs), m_rhs(rhs) {}

	std::string to_string() const final
	{
		return fmt::format("LESS_THAN       r{:<3} r{:<3} r{:<3}", m_dst, m_lhs, m_rhs);
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};


class BuildList final : public Instruction
{
	Register m_dst;
	std::vector<Register> m_srcs;

  public:
	BuildList(Register dst, std::vector<Register> srcs) : m_dst(dst), m_srcs(std::move(srcs)) {}

	std::string to_string() const final { return fmt::format("BUILD_LIST      r{:<3}", m_dst); }

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};


class BuildTuple final : public Instruction
{
	Register m_dst;
	std::vector<Register> m_srcs;

  public:
	BuildTuple(Register dst, std::vector<Register> srcs) : m_dst(dst), m_srcs(std::move(srcs)) {}

	std::string to_string() const final { return fmt::format("BUILD_TUPLE     r{:<3}", m_dst); }

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};

class BuildDict final : public Instruction
{
	Register m_dst;
	std::vector<Register> m_keys;
	std::vector<Register> m_values;

  public:
	BuildDict(Register dst, std::vector<Register> keys, std::vector<Register> values)
		: m_dst(dst), m_keys(std::move(keys)), m_values(std::move(values))
	{}

	std::string to_string() const final { return fmt::format("BUILD_DICT     r{:<3}", m_dst); }

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};


class GetIter final : public Instruction
{
	Register m_dst;
	Register m_src;

  public:
	GetIter(Register dst, Register src) : m_dst(dst), m_src(src) {}

	std::string to_string() const final
	{
		return fmt::format("GET_ITER        r{:<3} r{:<3}", m_dst, m_src);
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};

class ForIter final : public Instruction
{
	Register m_dst;
	Register m_src;
	std::shared_ptr<Label> m_exit_label;

  public:
	ForIter(Register dst, Register src, std::shared_ptr<Label> exit_label)
		: m_dst(dst), m_src(src), m_exit_label(std::move(exit_label))
	{}

	std::string to_string() const final
	{
		return fmt::format("FOR_ITER        r{:<3} r{:<3}", m_dst, m_src);
	}

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final;
};
