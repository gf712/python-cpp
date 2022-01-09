#include "GarbageCollector.hpp"
#include "interpreter/Interpreter.hpp"
#include "interpreter/InterpreterSession.hpp"
#include "memory/Heap.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyObject.hpp"
#include "runtime/PyType.hpp"
#include "vm/VM.hpp"

#include <csetjmp>

MarkSweepGC::MarkSweepGC() : m_frequency(1) {}

std::unordered_set<Cell *> MarkSweepGC::collect_roots() const
{
	if (!m_stack_bottom) {
		m_stack_bottom = bit_cast<uint8_t *>(Heap::the().start_stack_pointer());
	}
	std::unordered_set<Cell *> roots;

	// push register values onto the stack
	std::jmp_buf jump_buffer;
	setjmp(jump_buffer);

	// traverse the stack from the stack pointer to the stack origin/bottom
	uintptr_t *rsp_;
	asm volatile("movq %%rsp, %0" : "=r"(rsp_));

	uint8_t *rsp = static_cast<uint8_t *>(static_cast<void *>(rsp_));
	spdlog::trace("rsp={}, stack_bottom={}", (void *)rsp, (void *)m_stack_bottom);
	for (; rsp < m_stack_bottom; rsp += sizeof(uintptr_t)) {
		uint8_t *address =
			bit_cast<uint8_t *>(*bit_cast<uintptr_t *>(rsp)) - sizeof(GarbageCollected);
		spdlog::trace("checking address {}, pointer address={}", (void *)address, (void *)rsp);
		if (VirtualMachine::the().heap().slab().has_address(address)) {
			spdlog::trace("valid address {}", (void *)address);
			auto *cell = bit_cast<Cell *>(address + sizeof(GarbageCollected));
			if (cell->is_pyobject()) {
				auto *obj = static_cast<PyObject *>(cell);
				spdlog::trace("adding root {}@{}", obj->type()->name(), (void *)obj);
			}
			roots.insert(cell);
		}
	}

	const auto &registers = VirtualMachine::the().registers();
	if (registers.has_value()) {
		for (const auto &reg : registers->get()) {
			if (std::holds_alternative<PyObject *>(reg)) {
				if (auto *obj = std::get<PyObject *>(reg)) roots.insert(obj);
			}
		}
	}

	for (const auto &interpreter : VirtualMachine::the().interpreter_session()->interpreters()) {
		auto *execution_frame = interpreter->execution_frame();
		if (execution_frame) { roots.insert(execution_frame); }
		for (const auto &module: interpreter->get_available_modules()) {
			ASSERT(module)
			roots.insert(module);
		}
	}

	return roots;
}

struct MarkGCVisitor : Cell::Visitor
{
	std::unordered_set<Cell *> m_visited;
	void visit(Cell &cell)
	{
		// FIXME: due to the current awkward situation with static memory
		// 		  it can happen that
		if (m_visited.contains(&cell)) return;
		m_visited.insert(&cell);

		auto &heap = VirtualMachine::the().heap();
		uint8_t *cell_start = bit_cast<uint8_t *>(&cell);

		const bool static_memory =
			bit_cast<uintptr_t>(cell_start) >= bit_cast<uintptr_t>(heap.static_memory())
			&& bit_cast<uintptr_t>(cell_start)
				   < bit_cast<uintptr_t>(heap.static_memory() + heap.static_memory_size());

		if (!static_memory) {
			auto *obj_header = bit_cast<GarbageCollected *>(cell_start - sizeof(GarbageCollected));

			if (obj_header->black()) {
				if (cell.is_pyobject()) {
					spdlog::trace("Already visited {}@{}, skipping",
						static_cast<PyObject *>(&cell)->type_prototype().__name__,
						(void *)&cell);
				}
				return;
			}
			if (cell.is_pyobject()) {
				auto *obj = static_cast<PyObject *>(&cell);
				spdlog::trace("Visiting {}@{}", obj->type_prototype().__name__, (void *)&obj);
			}
			obj_header->mark(GarbageCollected::Color::BLACK);
		}

		cell.visit_graph(*this);
	}
};


void MarkSweepGC::mark_all_cell_unreachable(Heap &heap) const
{
	// TODO: once the ideal block sizes are fixed there should be an iterator
	//       returning a list of all blocks
	std::array blocks = {
		std::reference_wrapper{ heap.slab().block_32() },
		std::reference_wrapper{ heap.slab().block_64() },
		std::reference_wrapper{ heap.slab().block_128() },
		std::reference_wrapper{ heap.slab().block_256() },
		std::reference_wrapper{ heap.slab().block_512() },
		std::reference_wrapper{ heap.slab().block_1024() },
		std::reference_wrapper{ heap.slab().block_2048() },
	};
	for (const auto &block : blocks) {
		for (auto &chunk : block.get()->chunks()) {
			chunk.for_each_cell([](uint8_t *memory) {
				auto *header = static_cast<GarbageCollected *>(static_cast<void *>(memory));
				header->mark(GarbageCollected::Color::WHITE);
			});
		}
	}
}


void MarkSweepGC::mark_all_live_objects(const std::unordered_set<Cell *> &roots) const
{
	auto mark_visitor = std::make_unique<MarkGCVisitor>();

	// mark all live objects
	for (auto *root : roots) {
		spdlog::trace("Visiting root {}", (void *)root);
		root->visit_graph(*mark_visitor);
	}
}


void MarkSweepGC::sweep(Heap &heap) const
{
	// TODO: once the ideal block sizes are fixed there should be an iterator
	//       returning a list of all blocks
	std::array blocks = {
		std::reference_wrapper{ heap.slab().block_32() },
		std::reference_wrapper{ heap.slab().block_64() },
		std::reference_wrapper{ heap.slab().block_128() },
		std::reference_wrapper{ heap.slab().block_256() },
		std::reference_wrapper{ heap.slab().block_512() },
		std::reference_wrapper{ heap.slab().block_1024() },
		std::reference_wrapper{ heap.slab().block_2048() },
	};
	// sweep all the dead objects
	for (const auto &block : blocks) {
		for (auto &chunk : block.get()->chunks()) {
			chunk.for_each_cell_alive([&chunk](uint8_t *memory) {
				auto *header = static_cast<GarbageCollected *>(static_cast<void *>(memory));
				if (header->white()) {
					auto *cell = bit_cast<Cell *>(memory + sizeof(GarbageCollected));
					if (cell->is_pyobject()) {
						auto *obj = static_cast<PyObject *>(cell);
						spdlog::trace("Deallocating {}@{}", obj->type()->name(), (void *)obj);
					}
					spdlog::trace("Calling destructor of object at {}", (void *)cell);
					cell->~Cell();
					chunk.deallocate(memory);
				}
			});
		}
	}
}


void MarkSweepGC::run(Heap &heap) const
{
	if (m_pause) { return; }
	if (m_iterations_since_last_sweep++ < m_frequency) { return; }

	mark_all_cell_unreachable(heap);

	const auto roots = collect_roots();

	mark_all_live_objects(roots);

	sweep(heap);

	m_iterations_since_last_sweep = 0;
}

void MarkSweepGC::resume() { m_pause = false; }
void MarkSweepGC::pause() { m_pause = true; }