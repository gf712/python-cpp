#include "Instructions.hpp"


class RaiseVarargs final : public Instruction
{
	Register m_assertion;
	std::vector<Register> m_args;

  public:
	RaiseVarargs(Register assertion) : m_assertion(assertion) {}

	template<typename... Args>
	RaiseVarargs(Register assertion, Args &&... args)
		: m_assertion(assertion), m_args{std::forward<Args>(args)...}
	{}

	std::string to_string() const final
	{
		if (m_args.empty()) {
			return fmt::format("RAISE_VARARGS   r{:<3}", m_assertion);
		} else {
			// FIXME: should print out all registers, not just the first
			return fmt::format("RAISE_VARARGS   r{:<3} r{:<3}", m_assertion, m_args[0]);
		}
	}

	void execute(VirtualMachine &vm, Interpreter &interpreter) const final;

	void relocate(codegen::BytecodeGenerator &, size_t) final {}
};
