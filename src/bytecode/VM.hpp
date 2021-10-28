#pragma once

#include "forward.hpp"
#include "utilities.hpp"
#include "Heap.hpp"
#include "runtime/Value.hpp"

class VirtualMachine;

struct LocalFrame
{
	VirtualMachine *vm;
	LocalFrame(size_t frame_size, VirtualMachine *);
	LocalFrame(const LocalFrame &) = delete;
	LocalFrame(LocalFrame &&);
	~LocalFrame();
};

class VirtualMachine
	: NonCopyable
	, NonMoveable
{
	std::shared_ptr<Bytecode> m_bytecode;
	std::unique_ptr<Interpreter> m_interpreter;
	std::vector<std::vector<Value>> m_local_registers;
	size_t m_instruction_pointer{ 0 };
	Heap &m_heap;

	friend LocalFrame;

  public:
	VirtualMachine &create(std::shared_ptr<Bytecode> generator)
	{
		auto &vm = VirtualMachine::the();
		vm.push_generator(generator);
		return vm;
	}

	std::shared_ptr<PyObject> execute_statement(std::shared_ptr<Bytecode> bytecode);

	void push_generator(std::shared_ptr<Bytecode> generator);

	static VirtualMachine &the()
	{
		static auto *vm = new VirtualMachine();
		return *vm;
	}

	Value &reg(size_t idx) { return m_local_registers.back()[idx]; }
	const Value &reg(size_t idx) const { return m_local_registers.back()[idx]; }

	std::vector<Value> &registers() { return m_local_registers.back(); }
	const std::vector<Value> &registers() const { return m_local_registers.back(); }

	Heap &heap() { return m_heap; }

	std::unique_ptr<Interpreter> &interpreter() { return m_interpreter; }
	const std::unique_ptr<Interpreter> &interpreter() const { return m_interpreter; }

	size_t function_offset(size_t func_id);
	size_t function_register_count(size_t func_id);

	void set_instruction_pointer(size_t pos) { m_instruction_pointer = pos; }
	size_t instruction_pointer() const { return m_instruction_pointer; }

	void clear();

	int execute();
	int execute_frame();

	void dump() const;

	LocalFrame enter_frame(size_t frame_size) { return LocalFrame{ frame_size, this }; }

  private:
	VirtualMachine();
	VirtualMachine(std::unique_ptr<BytecodeGenerator> &&generator);

	void show_current_instruction(size_t index, size_t window) const;

	void push_frame(size_t frame_size) { m_local_registers.emplace_back(frame_size, nullptr); }

	void pop_frame() { m_local_registers.pop_back(); }
};
