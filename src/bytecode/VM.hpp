#pragma once

#include "forward.hpp"
#include "utilities.hpp"
#include "runtime/Value.hpp"
#include <memory>

static constexpr size_t KB = 1024;
static constexpr size_t MB = 1024 * KB;

class Heap
	: NonCopyable
	, NonMoveable
{
	uint8_t *m_memory;
	uint8_t *m_static_memory;
	size_t m_memory_size{ 500 * MB };
	size_t m_static_memory_size{ 4 * KB };
	size_t m_offset{ 0 };
	size_t m_static_offset{ 0 };

  public:
	static Heap &the()
	{
		static auto heap = Heap();
		return heap;
	}

	~Heap()
	{
		free(m_memory);
		free(m_static_memory);
		m_memory = nullptr;
		m_static_memory = nullptr;
	}

	void reset()
	{
		memset(m_memory, 0, m_memory_size);
		m_offset = 0;
	}

	template<typename T, typename... Args> std::shared_ptr<T> allocate(Args &&... args)
	{
		if (m_offset + sizeof(T) >= m_memory_size) { TODO(); }
		T *ptr = new (m_memory + m_offset) T(std::forward<Args>(args)...);
		m_offset += sizeof(T);
		return std::shared_ptr<T>(ptr, [](T *) { return; });
	}

	template<typename T, typename... Args> std::shared_ptr<T> allocate_static(Args &&... args)
	{
		if (m_static_offset + sizeof(T) >= m_static_memory_size) { TODO(); }
		T *ptr = new (m_static_memory + m_static_offset) T(std::forward<Args>(args)...);
		m_static_offset += sizeof(T);
		return std::shared_ptr<T>(ptr, [](T *) { return; });
	}

  private:
	Heap()
	{
		m_memory = static_cast<uint8_t *>(malloc(m_memory_size));
		m_static_memory = static_cast<uint8_t *>(malloc(m_static_memory_size));
	}
};


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
