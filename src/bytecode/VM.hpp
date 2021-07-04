#pragma once

#include "forward.hpp"
#include "runtime/Value.hpp"
#include <memory>

static constexpr size_t KB = 1024;

class Heap
{
	uint8_t *m_memory;
	size_t m_memory_size;
	size_t m_offset{ 0 };

  public:
	static Heap &the()
	{
		static auto heap = Heap();
		return heap;
	}

	~Heap()
	{
		free(m_memory);
		m_memory = nullptr;
	}

	void reset()
	{
		memset(m_memory, 0, m_memory_size);
		m_offset = 0;
	}

	template<typename T, typename... Args> std::shared_ptr<T> allocate(Args &&... args)
	{
		if (m_offset + sizeof(T) >= m_memory_size) { return nullptr; }
		T *ptr = new (m_memory + m_offset) T(std::forward<Args>(args)...);
		m_offset += sizeof(T);
		return std::shared_ptr<T>(ptr, [](T *) { return; });
	}

  private:
	Heap()
	{
		m_memory = static_cast<uint8_t *>(malloc(64 * KB));
		m_memory_size = 64 * KB;
	}
};

class VirtualMachine
{
	std::shared_ptr<Bytecode> m_bytecode;
	std::unique_ptr<Interpreter> m_interpreter;
	std::vector<Value> m_registers;
	size_t m_instruction_pointer{ 0 };
	size_t m_return_address{ 0 };
	Heap &m_heap;

  public:
	VirtualMachine &create(std::shared_ptr<Bytecode> generator)
	{
		auto &vm = VirtualMachine::the();
		vm.push_generator(generator);
		return vm;
	}

	void push_generator(std::shared_ptr<Bytecode> generator);

	static VirtualMachine &the()
	{
		static auto *vm = new VirtualMachine();
		return *vm;
	}

	Value &reg(size_t idx) { return m_registers[idx]; }
	const Value &reg(size_t idx) const { return m_registers[idx]; }

	std::vector<Value> &registers() { return m_registers; }
	const std::vector<Value> &registers() const { return m_registers; }

	Heap &heap() { return m_heap; }

	std::unique_ptr<Interpreter> &interpreter() { return m_interpreter; }
	const std::unique_ptr<Interpreter> &interpreter() const { return m_interpreter; }

	size_t function_offset(size_t func_id);

	void set_instruction_pointer(size_t pos) { m_instruction_pointer = pos; }
	size_t instruction_pointer() const { return m_instruction_pointer; }

	void set_return_address(size_t pos) { m_return_address = pos; }
	size_t return_address() const { return m_return_address; }

	void clear();

	int execute();

	void dump() const;

  private:
	VirtualMachine();
	VirtualMachine(std::unique_ptr<BytecodeGenerator> &&generator);
};
