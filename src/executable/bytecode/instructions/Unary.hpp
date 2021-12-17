#pragma once

#include "Instructions.hpp"


class UnaryPositive final : public Instruction
{
	Register m_destination;
	Register m_source;

  public:
	UnaryPositive(Register destination, Register source)
		: m_destination(destination), m_source(source)
	{}
	std::string to_string() const final
	{
		return fmt::format("UNARY_POSITIVE  r{:<3} r{:<3}", m_destination, m_source);
	}

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};


class UnaryNegative final : public Instruction
{
	Register m_destination;
	Register m_source;

  public:
	UnaryNegative(Register destination, Register source)
		: m_destination(destination), m_source(source)
	{}
	std::string to_string() const final
	{
		return fmt::format("UNARY_NEGATIVE  r{:<3} r{:<3}", m_destination, m_source);
	}

	void execute(VirtualMachine &, Interpreter &) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};


class UnaryInvert final : public Instruction
{
	Register m_destination;
	Register m_source;

  public:
	UnaryInvert(Register destination, Register source)
		: m_destination(destination), m_source(source)
	{}
	std::string to_string() const final
	{
		return fmt::format("UNARY_INVERT    r{:<3} r{:<3}", m_destination, m_source);
	}

	void execute(VirtualMachine &, Interpreter &) const final { TODO(); }

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};

class UnaryNot final : public Instruction
{
	Register m_destination;
	Register m_source;

  public:
	UnaryNot(Register destination, Register source) : m_destination(destination), m_source(source)
	{}
	std::string to_string() const final
	{
		return fmt::format("UNARY_NOT       r{:<3} r{:<3}", m_destination, m_source);
	}

	void execute(VirtualMachine &, Interpreter &) const final { TODO(); }

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};