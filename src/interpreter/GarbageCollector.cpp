#include "GarbageCollector.hpp"
#include "bytecode/Heap.hpp"

#include "bytecode/VM.hpp"
#include "Interpreter.hpp"
#include "runtime/PyDict.hpp"
#include "runtime/PyModule.hpp"
#include "runtime/PyObject.hpp"

#include <csetjmp>

MarkSweepGC::MarkSweepGC() : m_frequency(1) {}

std::unordered_set<Cell *> MarkSweepGC::collect_roots() const
{
	if (!m_stack_top) {
		m_stack_top = reinterpret_cast<uint8_t *>(Heap::the().start_stack_pointer());
	}
	std::unordered_set<Cell *> roots;

	// push register values onto the stack
	std::jmp_buf jump_buffer;
	setjmp(jump_buffer);

	// traverse the stack from the stack pointer to the top
	uintptr_t *rsp_;
	asm volatile("movq %%rsp, %0" : "=r"(rsp_));

	uint8_t *rsp = static_cast<uint8_t *>(static_cast<void *>(rsp_));
	for (; rsp < m_stack_top; rsp += sizeof(uintptr_t)) {
		uint8_t *address = reinterpret_cast<uint8_t *>(*reinterpret_cast<uintptr_t *>(rsp))
						   - sizeof(GarbageCollected);
		spdlog::debug("checking address {}, pointer address={}", (void *)address, (void *)rsp);
		if (VirtualMachine::the().heap().slab().has_address(address)) {
			spdlog::debug("valid address {}", (void *)address);
			auto *cell = bit_cast<Cell *>(address + sizeof(GarbageCollected));
			if (cell->is_pyobject()) {
				auto *obj = static_cast<PyObject *>(cell);
				spdlog::debug("adding root {}@{}", object_name(obj->type()), (void *)obj);
			}
			roots.insert(cell);
		}
	}

	for (const auto &reg : VirtualMachine::the().registers()) {
		if (std::holds_alternative<PyObject *>(reg)) {
			if (auto *obj = std::get<PyObject *>(reg)) roots.insert(obj);
		}
	}

	auto *execution_frame = VirtualMachine::the().interpreter()->execution_frame();
	if (execution_frame) { roots.insert(execution_frame); }

	return roots;
}

struct MarkGCVisitor : Cell::Visitor
{
	void visit(Cell &cell)
	{
		uint8_t *cell_start = static_cast<uint8_t *>(static_cast<void *>(&cell));
		auto *obj_header = static_cast<GarbageCollected *>(
			static_cast<void *>(cell_start - sizeof(GarbageCollected)));

		if (obj_header->black()) {
			// spdlog::debug("Already visited {}@{}, skipping",
			// 	object_name(static_cast<PyObject *>(&cell)->type()),
			// 	(void *)&cell);
			return;
		}
		if (cell.is_pyobject()) {
			auto *obj = static_cast<PyObject *>(&cell);
			// spdlog::debug("object {} address @{}",
			// 	object_name(static_cast<PyObject *>(obj)->type()),
			// 	(void *)obj);
			spdlog::debug("Visiting {}@{}", object_name(obj->type()), (void *)&obj);
		}

		obj_header->mark(GarbageCollected::Color::BLACK);
		cell.visit_graph(*this);
	}
};


void MarkSweepGC::run(Heap &heap) const
{
	if (m_iterations_since_last_sweep++ < m_frequency) { return; }

	// TODO: once the ideal block sizes are fixed there should be an iterator
	//       returning a list of all blocks
	std::array blocks = { std::reference_wrapper{ heap.slab().block_512() },
		std::reference_wrapper{ heap.slab().block_1024() } };

	// {// mark all cells as unreachable
	// 	const auto &block = heap.slab().block();
	// 	for (auto &chunk : block->chunks()) {
	// 		chunk.for_each_cell([](uint8_t *memory) {
	// 			auto *header = static_cast<GarbageCollected<Cell> *>(static_cast<void *>(memory));
	// 			header->mark(GarbageCollected<Cell>::Color::WHITE);
	// 		});
	// 	}
	// }

	// collect roots
	const auto roots = collect_roots();

	auto mark_visitor = std::make_unique<MarkGCVisitor>();

	// mark all live objects
	for (auto *root : roots) {
		// spdlog::debug("Visiting root {}@{}",
		// 	object_name(static_cast<PyObject *>(root)->type()),
		// 	(void *)&root);
		root->visit_graph(*mark_visitor);
	}

	// sweep all the dead objects
	for (const auto &block : blocks) {
		for (auto &chunk : block.get()->chunks()) {
			chunk.for_each_cell_alive([&chunk](uint8_t *memory) {
				auto *header = static_cast<GarbageCollected *>(static_cast<void *>(memory));
				if (header->white()) {
					auto *cell = bit_cast<Cell *>(memory + sizeof(GarbageCollected));
					if (cell->is_pyobject()) {
						auto *obj = static_cast<PyObject *>(cell);
						spdlog::debug("Deallocating {}@{}", object_name(obj->type()), (void *)obj);
					}
					// header->~GarbageCollected();
					header->mark(GarbageCollected::Color::WHITE);
					cell->~Cell();
					chunk.deallocate(memory);
				}
			});
		}
	}

	m_iterations_since_last_sweep = 0;
}
